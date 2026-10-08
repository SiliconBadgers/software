"""Hardware/assumption configuration: defaults, overrides, validation, dotted-path access, stable hash.

Every number here is an ASSUMPTION unless docs/ASSUMPTIONS.md says it is measured. Defaults reproduce the
"balanced" design saved by the 2026-09-24 explorer (configs/recommended-balanced.json) wherever a matching
parameter exists, so results are comparable to that experiment; parameters that did not exist there are new.

The `legacy` switches (matrix.tile_mode, memory.residency, memory.l1_feed, memory.fusion, schedule.mode,
matrix.pipelined_tiles) select the older equation for that piece so each change can be ablated one at a time.
SWITCHES below is the register of every modelling switch: what the explorer did, which setting is the
pessimistic bracket, and what the other setting assumes. Results carry it so no switch is in effect unstated.
"""
import copy
import hashlib
import json
from pathlib import Path

DEFAULTS = {
    "clock_mhz": 300,
    "batch": 1,
    "launch_cycles": 40,
    "sync_cycles": 100,          # cross-unit hand-off (new); placeholder
    "matrix": {
        "count": 4, "rows": 32, "cols": 32, "efficiency": 0.75,
        "caps": ["gemm", "gemv", "attention", "requant"],
        # MACs per PE per cycle by weight-format class (formats.rate_class); "aa" = activation x activation
        "rates": {"w4": 2.0, "w5": 1.0, "w6": 1.0, "w8": 1.0, "w16": 0.5, "w32": 0.25, "aa": 0.5},
        "pipelined_tiles": True,     # True: array fill/drain once per op. False (legacy): once per output tile
        "tile_mode": "search",       # "search" | "fixed" (legacy: tileN/tileK heuristic)
        "tile_k": 256, "conv_penalty": 4.0,
        "legacy_tile_n": 64,
    },
    "vector": {"count": 4, "lanes": 32, "efficiency": 0.7, "special_cycles": 8,
               "caps": ["elementwise", "reduce", "exp", "rope", "quant", "recurrent", "conv"]},
    "recurrent": {"count": 2, "lanes": 128, "efficiency": 0.7, "scratch_kib": 64,
                  # "serial" | "chunked" | "auto" (cheaper of serial and chunked) | "matrix" (the captured serial
                  # update run on the matrix arrays at matrix_penalty; needs the gemm capability; never chosen
                  # by "auto")
                  "lowering": "auto",
                  "chunk": 64,
                  # W8xA16-equivalent PE cycles per 32-bit recurrence MAC for lowering = "matrix". Not established
                  # hardware performance: the 2026-10-03 recurrence study sweeps 1x-32x and reports 4x and 16x
                  "matrix_penalty": 16.0},
    "dma": {"count": 2, "bytes_per_cycle": 64},
    "scalar": {"count": 1, "ipc": 1.0, "mhz": 300, "cycles_per_op": 4, "cycles_per_special": 40},
    "host": {"enabled": False, "link_gbps": 16.0, "link_latency_us": 20.0, "use_measured_cpu_time": False},
    "l1": {"mib": 8, "banks": 64, "bytes_per_bank_cycle": 16, "efficiency": 0.65,
           "state_reserve": 0.25, "tile_fraction": 0.5},
    "hbm": {"gib": 16, "gbs": 460, "efficiency": 0.7},
    "precision": {"weights": "captured",   # "captured" | "uniform_int4" | "uniform_int8"
                  "group_size": 64, "scale_bytes": 2, "act_bits": 16, "kv_bits": 16, "state_bits": 32},
    "attention": {"causal_skip": True, "kv_padding": "captured", "kv_tile": 128, "overlap_softmax": True,
                  "mask_onchip": True, "capacity_tokens": None},
    "memory": {"residency": "belady",      # "belady" | "per_op" (legacy spill proxy)
               "fusion": "chains",         # "chains" | "groups" (anchored groups, groups.py) | "none"
               "l1_feed": "array",         # "array" | "tile" (legacy)
               # "via_l1": HBM -> L1 -> array (explorer: no direct inter-unit channels), so every weight byte
               # crosses L1 twice. "direct": weights stream HBM -> array staging registers and skip L1 (needs
               # per-PE weight buffers that this model does not size or cost; an architecture decision, not a given)
               "weight_path": "via_l1"},
    "schedule": {"mode": "pools",          # "pools" (list schedule, reported with bounds) | "serial" (legacy sum)
                 "within_op_overlap": True},
    "resources": {"dsp_budget": 9024, "shell_reserve": 0.2, "dsp_per_pe": 1.0, "dsp_per_vector_lane": 0.5,
                  "dsp_per_recurrent_lane": 1.0,
                  "cost": {"matrix_pe": 1.0, "vector_lane": 8.0, "recurrent_lane": 10.0, "sram_mib": 128.0,
                           "capability": 40.0, "scalar_core": 200.0, "dma_engine": 40.0,
                           # new: the explorer priced SRAM by capacity only, so extra banks (L1 bandwidth) and a direct
                           # weight path were free. Placeholders that make those trade-offs visible; 0 restores "free".
                           "l1_bank": 8.0, "direct_weight_buffer_per_pe": 0.25}},
}

_ENUMS = {
    ("matrix", "tile_mode"): {"search", "fixed"},
    ("recurrent", "lowering"): {"serial", "chunked", "auto", "matrix"},
    ("precision", "weights"): {"captured", "uniform_int4", "uniform_int8"},
    ("attention", "kv_padding"): {"captured", "exact"},
    ("memory", "residency"): {"belady", "per_op"},
    ("memory", "fusion"): {"chains", "groups", "none"},
    ("memory", "l1_feed"): {"array", "tile"},
    ("memory", "weight_path"): {"via_l1", "direct"},
    ("schedule", "mode"): {"pools", "serial"},
}
_POSITIVE_INT = [("batch",), ("matrix", "rows"), ("matrix", "cols"), ("vector", "lanes"), ("recurrent", "lanes"),
                 ("recurrent", "chunk"), ("l1", "banks"), ("l1", "bytes_per_bank_cycle"), ("precision", "group_size"),
                 ("attention", "kv_tile"), ("matrix", "tile_k"), ("matrix", "legacy_tile_n")]
_COUNTS = [("matrix", "count"), ("vector", "count"), ("recurrent", "count"), ("dma", "count"), ("scalar", "count")]
_FRACTIONS = [("matrix", "efficiency"), ("vector", "efficiency"), ("recurrent", "efficiency"), ("l1", "efficiency"),
              ("hbm", "efficiency")]


# Every modelling switch. `explorer` is the 2026-09-24 explorer's behaviour (None: it had no equivalent).
# `conservative` is the pessimistic bracket, or None when neither setting is the cautious one (a derived
# accounting change or a design choice). `status` follows docs/EQUATIONS.md. `alternative` marks a switch that
# is deliberately left out of the bracket but whose other setting is still reported, alone and as an upper bound.
SWITCHES = [
    {"path": "precision.weights", "explorer": "uniform_int8", "conservative": None, "status": "design choice",
     "assumes": "captured: weight bytes are the mixed Q4_K_M sizes in the capture; per-format MAC rates are assumed"},
    {"path": "matrix.tile_mode", "explorer": "fixed", "conservative": None, "status": "derived",
     "assumes": "search: the tile shape that minimises operand re-streaming within the same L1 footprint"},
    {"path": "matrix.pipelined_tiles", "explorer": False, "conservative": False, "status": "assumption",
     "assumes": "true: array fill/drain is paid once per op (double-buffered accumulators), not once per tile"},
    {"path": "memory.l1_feed", "explorer": "tile", "conservative": None, "status": "derived",
     "assumes": "array: every output tile streams both operands past the PEs, weight scale bytes included"},
    {"path": "memory.residency", "explorer": "per_op", "conservative": None, "status": "derived + assumption",
     "assumes": "belady: activations stay in L1 while they fit and are needed; the static L1 partition is assumed"},
    {"path": "memory.fusion", "explorer": "none", "conservative": "none", "status": "assumption",
     "assumes": "chains / groups: the units apply fused consumers in place, so the intermediate is never stored"},
    {"path": "memory.weight_path", "explorer": "via_l1", "conservative": "via_l1", "status": "assumption",
     "assumes": "direct: weights stream HBM -> per-PE staging and skip L1; the buffers are not sized here"},
    {"path": "recurrent.lowering", "explorer": "serial", "conservative": "serial", "status": "assumption",
     "assumes": "auto / chunked: a chunk-parallel lowering nobody has implemented; matrix: the matrix arrays run "
                "the 32-bit update at recurrent.matrix_penalty"},
    {"path": "attention.causal_skip", "explorer": None, "conservative": False, "status": "derived + assumption",
     "assumes": "true: fused attention skips fully masked key tiles (flash-attention-on graphs only)"},
    {"path": "attention.overlap_softmax", "explorer": None, "conservative": False, "status": "assumption",
     "assumes": "true: softmax runs on the vector units behind the matrix work instead of after it"},
    {"path": "attention.mask_onchip", "explorer": None, "conservative": False, "status": "assumption",
     "assumes": "true: the causal mask is generated on-chip instead of read from HBM"},
    {"path": "attention.kv_padding", "explorer": None, "conservative": "captured", "status": "assumption",
     "assumes": "exact: fused attention reads only the live keys instead of the capture's padded KV length"},
    {"path": "schedule.mode", "explorer": "serial", "conservative": "serial", "status": "derived + assumption",
     "assumes": "pools: ops on different unit pools overlap once their captured dependencies are met"},
    # Not bracketed: max(compute, L1, HBM) is the explorer's own node-duration rule and the baseline of every
    # sensitivity study in research/compute-mapping, so the pessimistic comparison keeps it to stay comparable with
    # them. Off, the three are charged in sequence, as if a unit could not compute while its operands stream. That
    # is an upper bound rather than a bracket around this engine's changes, so it is reported separately.
    {"path": "schedule.within_op_overlap", "explorer": True, "conservative": None, "alternative": False,
     "status": "assumption",
     "assumes": "true: compute, L1 and HBM service overlap inside one op. This is the explorer's node-duration rule "
                "and every study's baseline, so it is not bracketed; false (the three in sequence) is reported "
                "separately as an upper bound"},
]


def switch_report(cfg):
    """Where each modelling switch stands in this config. `optimistic` marks a switch that is off its pessimistic
    bracket, i.e. one whose assumption the reported numbers depend on. `informational` marks an unbracketed switch
    whose alternative setting is still worth reporting (see SWITCHES)."""
    rows = []
    for s in SWITCHES:
        value = get_path(cfg, s["path"])
        alternative = s.get("alternative")
        rows.append({**s, "alternative": alternative, "value": value,
                     "differs_from_explorer": s["explorer"] is not None and value != s["explorer"],
                     "optimistic": s["conservative"] is not None and value != s["conservative"],
                     "informational": alternative is not None and value != alternative})
    return rows


def conservative(cfg, upper_bound=False):
    """A copy of cfg with every switch that has a pessimistic bracket set to it. Hardware parameters are untouched.
    With upper_bound, the unbracketed switches that have an `alternative` are moved to it as well."""
    out = copy.deepcopy(cfg)
    for s in SWITCHES:
        if s["conservative"] is not None:
            set_path(out, s["path"], s["conservative"])
        elif upper_bound and s.get("alternative") is not None:
            set_path(out, s["path"], s["alternative"])
    return out


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_path(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def set_path(cfg, dotted, value):
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur[part]
    if parts[-1] not in cur:
        raise KeyError(f"unknown config path {dotted!r}")
    cur[parts[-1]] = value


def validate(cfg):
    """Return a list of human-readable problems (empty when the config is usable)."""
    errors = []

    def num(path, positive=False):
        try:
            v = get_path(cfg, ".".join(path))
        except KeyError:
            errors.append(f"missing {'.'.join(path)}")
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
            errors.append(f"{'.'.join(path)} must be a finite number")
            return None
        if positive and v <= 0:
            errors.append(f"{'.'.join(path)} must be > 0")
        if not positive and v < 0:
            errors.append(f"{'.'.join(path)} must be >= 0")
        return v

    for path in _POSITIVE_INT:
        v = num(path, True)
        if v is not None and int(v) != v:
            errors.append(f"{'.'.join(path)} must be an integer")
    for path in _COUNTS:
        v = num(path)
        if v is not None and int(v) != v:
            errors.append(f"{'.'.join(path)} must be an integer")
    for path in _FRACTIONS:
        v = num(path, True)
        if v is not None and v > 1:
            errors.append(f"{'.'.join(path)} must be in (0, 1]")
    for path in [("clock_mhz",), ("hbm", "gib"), ("hbm", "gbs"), ("l1", "mib"), ("scalar", "ipc"), ("scalar", "mhz"),
                 ("scalar", "cycles_per_op"), ("recurrent", "matrix_penalty")]:
        num(path, True)
    for path in [("launch_cycles",), ("sync_cycles",), ("vector", "special_cycles"), ("scalar", "cycles_per_special"),
                 ("recurrent", "scratch_kib"), ("dma", "bytes_per_cycle"), ("host", "link_gbps"),
                 ("host", "link_latency_us"), ("matrix", "conv_penalty")]:
        num(path)
    for name in ("state_reserve", "tile_fraction"):
        v = num(("l1", name))
        if v is not None and v >= 1:
            errors.append(f"l1.{name} must be < 1")
    try:
        if cfg["l1"]["state_reserve"] + cfg["l1"]["tile_fraction"] >= 1:
            errors.append("l1.state_reserve + l1.tile_fraction must leave room for activation residency (< 1)")
    except (KeyError, TypeError):
        pass
    for (section, key), allowed in _ENUMS.items():
        v = cfg.get(section, {}).get(key)
        if v not in allowed:
            errors.append(f"{section}.{key} must be one of {sorted(allowed)}")
    for section, key in [("matrix", "caps"), ("vector", "caps")]:
        v = cfg.get(section, {}).get(key)
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
            errors.append(f"{section}.{key} must be a list of strings")
    for bits in ("act_bits", "kv_bits", "state_bits"):
        if cfg["precision"].get(bits) not in (8, 16, 32):
            errors.append(f"precision.{bits} must be 8, 16 or 32")
    return errors


def make_config(overrides=None, base=None):
    """Merge overrides (nested dict) into base (default DEFAULTS) and validate. Raises ValueError on problems."""
    cfg = deep_merge(DEFAULTS if base is None else base, overrides or {})
    problems = validate(cfg)
    if problems:
        raise ValueError("invalid config: " + "; ".join(problems))
    return cfg


def apply_sets(cfg, sets):
    """Apply ['a.b=3', 'c.d=[1,2]'] style assignments (values parsed as JSON, else kept as strings)."""
    cfg = copy.deepcopy(cfg)
    for item in sets or []:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        set_path(cfg, key.strip(), value)
    problems = validate(cfg)
    if problems:
        raise ValueError("invalid config: " + "; ".join(problems))
    return cfg


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


PROFILE_DIR = Path(__file__).resolve().parents[1] / "profiles"


def load_profile(path_or_name):
    """Load profiles/<name>.json (or an explicit path); returns (config, profile document)."""
    path = Path(path_or_name)
    if not path.is_file():
        path = PROFILE_DIR / (str(path_or_name) if str(path_or_name).endswith(".json") else f"{path_or_name}.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return make_config(doc.get("config")), doc
