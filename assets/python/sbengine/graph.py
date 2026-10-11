"""Load a captured scheduled graph (prefill.json / decode.json, or their .json.gz copies) into typed nodes.

Everything here is derived from the capture: shapes, dtypes, byte sizes, operand slots, view chains and the
scheduler order. Nothing is extrapolated. Conventions follow core/analysis/graph.py: GGML shapes list the
innermost dimension first, and `macs_of` is the same MAC count that module reports (checked in tests).
"""
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

METADATA_OPS = {"NONE", "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"}
ELEMENTWISE_OPS = {"MUL", "ADD", "SCALE"}
ACTIVATION_OPS = {"SILU", "SIGMOID", "SOFTPLUS", "SWIGLU"}
MEMORY_OPS = {"GET_ROWS", "SET_ROWS", "CPY", "CONT", "CONCAT"}
KNOWN_OPS = ({"MUL_MAT", "FLASH_ATTN_EXT", "GATED_DELTA_NET", "SSM_CONV", "SOFT_MAX", "RMS_NORM", "ROPE"}
             | ELEMENTWISE_OPS | ACTIVATION_OPS | MEMORY_OPS)

_KV = re.compile(r"^cache_[kv]_l\d+$")
_STATE = re.compile(r"^cache_[rs]_l\d+$")
_LAYER = [re.compile(r"blk\.(\d+)\."), re.compile(r"-(\d+)(?:\s|$)")]


@dataclass(frozen=True)
class Tensor:
    id: int
    name: str
    op: str            # GGML op family (UNARY, GLU, ...)
    desc: str          # specific op (SILU, SWIGLU, ...): what the model keys on
    dtype: str
    shape: tuple
    elements: int
    span: int          # bytes spanned in storage
    nbytes: int        # logical bytes
    view_source: int
    weight: bool
    sources: tuple     # ((slot, tensor id), ...)

    @property
    def is_metadata(self):
        return self.desc in METADATA_OPS


@dataclass
class Node:
    idx: int
    tensor: Tensor
    srcs: tuple        # (Tensor, ...) unique, in slot order
    slots: dict        # slot -> Tensor
    dst_root: int
    src_roots: tuple   # root ids of srcs (same order)
    spos: int = -1     # position in the scheduler order (index into scheduled_nodes, views included)

    @property
    def op(self):
        return self.tensor.desc


def layer_of(name):
    for pattern in _LAYER:
        m = pattern.search(name)
        if m:
            return int(m.group(1))
    return None


def capture_path(directory, phase):
    """The captured graph for one phase: `{phase}.json` as the capture wrote it, else the gzipped
    `{phase}.json.gz` that a committed run keeps (results/README.md). None when neither exists."""
    for name in (f"{phase}.json", f"{phase}.json.gz"):
        path = Path(directory) / name
        if path.is_file():
            return path
    return None


def has_capture(directory):
    return capture_path(directory, "prefill") is not None and capture_path(directory, "decode") is not None


class Graph:
    """One phase (prefill or decode) of one captured workload."""

    def __init__(self, path, phase):
        path = Path(path)
        text = (gzip.decompress(path.read_bytes()).decode("utf-8") if path.suffix == ".gz"
                else path.read_text(encoding="utf-8"))
        raw = json.loads(text)
        self.path, self.phase, self.meta = path, phase, raw["metadata"]
        self.tensors = {}
        for t in raw["tensors"]:
            self.tensors[t["id"]] = Tensor(
                id=t["id"], name=t["name"], op=t["op"], desc=t["op_desc"], dtype=t["dtype"], shape=tuple(t["shape"]),
                elements=t["elements"], span=t["span_bytes"], nbytes=t["logical_bytes"], view_source=t["view_source"],
                weight=bool(t["weight_buffer"]), sources=tuple((s["slot"], s["tensor"]) for s in t["sources"]))
        self.order = list(raw["scheduled_nodes"])
        self._root = {}
        self.nodes = []
        self.metadata_nodes = self.zero_nodes = 0
        for spos, tid in enumerate(self.order):
            t = self.tensors[tid]
            if t.is_metadata:
                self.metadata_nodes += 1
                continue
            if t.elements == 0:
                self.zero_nodes += 1
                continue
            slots, seen = {}, {}
            for slot, sid in t.sources:
                slots[slot] = self.tensors[sid]
                seen.setdefault(sid, self.tensors[sid])
            srcs = tuple(seen.values())
            dst_root = self.root(tid)
            self.nodes.append(Node(len(self.nodes), t, srcs, slots, dst_root,
                                   tuple(self.root(s.id) for s in srcs), spos))
        self._index()

    # -- storage -------------------------------------------------------------------------------------------
    def root(self, tid):
        """Storage owner of a tensor: follow the view chain to the tensor that owns the bytes."""
        if tid not in self._root:
            chain, cur = [], tid
            while self.tensors[cur].view_source >= 0 and cur not in chain:
                chain.append(cur)
                cur = self.tensors[cur].view_source
            for c in chain:
                self._root[c] = cur
            self._root[cur] = cur
        return self._root[tid]

    def kind(self, root_id):
        """weight | kv | state | mask | input | activation"""
        t = self.tensors[root_id]
        if t.weight:
            return "weight"
        if _KV.match(t.name):
            return "kv"
        if _STATE.match(t.name):
            return "state"
        if t.desc == "NONE":
            return "mask" if "mask" in t.name else "input"
        return "activation"

    def roots_of_kind(self, kind):
        """Storage-owning tensors of one kind (weight | kv | state | ...)."""
        cache = self.__dict__.setdefault("_kind_roots", {})
        if kind not in cache:
            cache[kind] = [tid for tid, t in self.tensors.items() if t.view_source < 0 and self.kind(tid) == kind]
        return cache[kind]

    @property
    def matrix_weight_roots(self):
        """Weights consumed as a matmul operand or a gather table: the ones a weight-format policy applies to
        (norm vectors and the conv1d kernel keep their captured 32-bit storage)."""
        if not hasattr(self, "_mwr"):
            self._mwr = {self.root(n.slots[0].id) for n in self.nodes
                         if n.tensor.op in ("MUL_MAT", "GET_ROWS") and 0 in n.slots
                         and self.kind(self.root(n.slots[0].id)) == "weight"}
        return self._mwr

    def _index(self):
        self.producer, self.consumers, self.writers = {}, {}, {}
        for n in self.nodes:
            kind = self.kind(n.dst_root)
            if kind == "activation":
                self.producer[n.dst_root] = n.idx
            else:  # in-place write into a persistent tensor (SET_ROWS / CPY into a cache)
                self.writers.setdefault(n.dst_root, []).append(n.idx)
        for n in self.nodes:
            for r in n.src_roots:
                if r != n.dst_root:
                    self.consumers.setdefault(r, []).append(n.idx)

    def deps(self):
        """Per-node dependency list: producers of consumed activations, plus read-after-write,
        write-after-read and write-after-write ordering on persistent tensors (KV cache, recurrent/conv
        state), which plain tensor dataflow does not express. Ordering is whole-buffer, so it is conservative
        for writes that touch different rows."""
        deps = [set() for _ in self.nodes]
        last_write, readers = {}, {}
        for n in self.nodes:
            for r in n.src_roots:
                if r in self.producer and self.producer[r] != n.idx:
                    deps[n.idx].add(self.producer[r])
                if self.kind(r) in ("kv", "state") and r != n.dst_root:
                    if r in last_write:
                        deps[n.idx].add(last_write[r])
                    readers.setdefault(r, []).append(n.idx)
            if self.kind(n.dst_root) in ("kv", "state"):
                for reader in readers.get(n.dst_root, []):
                    if reader != n.idx:
                        deps[n.idx].add(reader)
                if n.dst_root in last_write:   # e.g. prefill clears a state buffer (SCALE) before the write-back (CPY)
                    deps[n.idx].add(last_write[n.dst_root])
                readers[n.dst_root] = []
                last_write[n.dst_root] = n.idx
        return [sorted(d) for d in deps]

    # -- work accounting (same counts as core/analysis/graph.py) ------------------------------------------
    @staticmethod
    def macs_of(node):
        t = node.tensor
        if t.op == "MUL_MAT":
            return t.elements * node.slots[0].shape[0]
        if t.op == "SSM_CONV":
            return t.elements * node.slots[1].shape[0]
        return None

    def totals(self):
        macs = {}
        for n in self.nodes:
            m = self.macs_of(n)
            if m is not None:
                macs[n.tensor.op] = macs.get(n.tensor.op, 0) + m
        return macs


@dataclass
class Workload:
    name: str
    directory: Path
    prefill: Graph
    decode: Graph

    @property
    def meta(self):
        return self.prefill.meta

    def graph(self, phase):
        return self.prefill if phase == "prefill" else self.decode


def load_workload(directory):
    directory = Path(directory)
    return Workload(directory.name, directory, Graph(capture_path(directory, "prefill"), "prefill"),
                    Graph(capture_path(directory, "decode"), "decode"))


def latest_run(results_dir):
    runs = []
    for manifest in Path(results_dir).glob("*/run-manifest.json"):
        if (manifest.parent / "graphs").is_dir():
            started = json.loads(manifest.read_text(encoding="utf-8")).get("run", {}).get("started_at", "")
            runs.append((started, manifest.parent))
    if not runs:
        raise SystemExit(f"No run with graphs/ under {results_dir}")
    return max(runs)[1]
