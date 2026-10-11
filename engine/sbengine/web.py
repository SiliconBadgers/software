"""Local web app: `python -m sbengine serve` -> http://127.0.0.1:8765

A stdlib-only HTTP server that serves one page (web/index.html) and a small JSON API over the real engine, so every
slider re-runs the actual model. It binds to the loopback interface by default, accepts only known routes, validates every
config through config.make_config, and never takes a file path from the client (workloads are chosen from a fixed list).
"""
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import VERSION, sweep
from .config import DEFAULTS, PROFILE_DIR, get_path, load_profile, make_config, validate
from .graph import has_capture, latest_run, load_workload
from .model import evaluate
from .trace import load_trace

PAGE = Path(__file__).with_name("web") / "index.html"
MAX_BODY = 1 << 20
MAX_SWEEP_POINTS = 80

# Control schema: the page builds itself from this, so config knowledge lives in one place.
# kind: num | int | select | bool | caps ; group order is display order.
_PCT = {"min": 0.05, "max": 1.0, "step": 0.05}
CONTROLS = [
    {"path": "clock_mhz", "label": "Clock", "unit": "MHz", "group": "Compute units", "kind": "num", "min": 50, "max": 1000, "step": 25,
     "help": "Accelerator clock. Every unit and the L1 scale with it."},
    {"path": "batch", "label": "Batch (independent sequences)", "group": "Compute units", "kind": "int", "min": 1, "max": 32, "step": 1,
     "help": "Extrapolated from the batch-1 capture: weights shared, activations/KV/state scale."},
    {"path": "matrix.count", "label": "Matrix units", "group": "Compute units", "kind": "int", "min": 0, "max": 16, "step": 1,
     "help": "0 forces every matmul onto the scalar fallback core (valid but very slow)."},
    {"path": "matrix.rows", "label": "Array rows", "group": "Compute units", "kind": "select", "options": [8, 16, 32, 64, 128]},
    {"path": "matrix.cols", "label": "Array columns", "group": "Compute units", "kind": "select", "options": [8, 16, 32, 64, 128]},
    {"path": "matrix.efficiency", "label": "Matrix efficiency", "group": "Compute units", "kind": "num", **_PCT,
     "help": "Assumption. RTL utilisation of the array would settle it."},
    {"path": "matrix.rates.w4", "label": "MACs/PE/cycle, 4-bit weights", "group": "Compute units", "kind": "num", "min": 0.25, "max": 8, "step": 0.25,
     "help": "q4_K: 67% of prefill MACs."},
    {"path": "matrix.rates.w5", "label": "MACs/PE/cycle, 5-bit weights", "group": "Compute units", "kind": "num", "min": 0.25, "max": 8, "step": 0.25,
     "help": "q5_K: 22% of prefill MACs."},
    {"path": "matrix.rates.w6", "label": "MACs/PE/cycle, 6-bit weights", "group": "Compute units", "kind": "num", "min": 0.25, "max": 8, "step": 0.25,
     "help": "q6_K: 11% of prefill MACs."},
    {"path": "matrix.rates.aa", "label": "MACs/PE/cycle, activation x activation", "group": "Compute units", "kind": "num", "min": 0.25, "max": 8, "step": 0.25},
    {"path": "matrix.caps", "label": "Matrix capabilities", "group": "Compute units", "kind": "caps",
     "options": ["gemm", "gemv", "attention", "requant", "conv"], "help": "Remove one to see the fallback cost (gemv, attention, requant)."},
    {"path": "matrix.pipelined_tiles", "label": "Fill/drain once per op", "group": "Compute units", "kind": "bool"},
    {"path": "vector.count", "label": "Vector units", "group": "Vector and recurrence", "kind": "int", "min": 0, "max": 16, "step": 1},
    {"path": "vector.lanes", "label": "Lanes per vector unit", "group": "Vector and recurrence", "kind": "select", "options": [8, 16, 32, 64, 128]},
    {"path": "vector.efficiency", "label": "Vector efficiency", "group": "Vector and recurrence", "kind": "num", **_PCT},
    {"path": "vector.special_cycles", "label": "exp / rsqrt cost", "unit": "cycles", "group": "Vector and recurrence", "kind": "int", "min": 1, "max": 64, "step": 1},
    {"path": "vector.caps", "label": "Vector capabilities", "group": "Vector and recurrence", "kind": "caps",
     "options": ["elementwise", "reduce", "exp", "rope", "quant", "recurrent", "conv", "gemm"], "help": "Remove exp to push softmax/SiLU to the scalar core."},
    {"path": "recurrent.count", "label": "Recurrence units", "group": "Vector and recurrence", "kind": "int", "min": 0, "max": 8, "step": 1,
     "help": "0 runs the recurrence on the vector units."},
    {"path": "recurrent.lanes", "label": "Lanes per recurrence unit", "group": "Vector and recurrence", "kind": "select", "options": [32, 64, 128, 256, 512]},
    {"path": "recurrent.efficiency", "label": "Recurrence efficiency", "group": "Vector and recurrence", "kind": "num", **_PCT},
    {"path": "recurrent.lowering", "label": "Recurrence lowering", "group": "Vector and recurrence", "kind": "select", "options": ["auto", "serial", "chunked", "matrix"],
     "help": "Largest single assumption: chunked runs the recurrence as matrix GEMMs and is ~8x cheaper in prefill."},
    {"path": "recurrent.matrix_penalty", "label": "Matrix recurrence penalty", "unit": "x", "group": "Vector and recurrence", "kind": "select", "options": [1, 2, 4, 8, 16, 32],
     "help": "Only for lowering = matrix: W8xA16-equivalent PE cycles per 32-bit recurrence MAC. Not established; 4x and 16x are the study's reference points."},
    {"path": "recurrent.chunk", "label": "Chunk size", "unit": "tokens", "group": "Vector and recurrence", "kind": "select", "options": [16, 32, 64, 128, 256]},
    {"path": "dma.count", "label": "DMA engines", "group": "Vector and recurrence", "kind": "int", "min": 0, "max": 8, "step": 1},
    {"path": "dma.bytes_per_cycle", "label": "DMA bytes/engine/cycle", "group": "Vector and recurrence", "kind": "select", "options": [16, 32, 64, 128, 256]},
    {"path": "l1.mib", "label": "L1 capacity", "unit": "MiB", "group": "Memory", "kind": "num", "min": 0.25, "max": 64, "step": 0.25},
    {"path": "l1.banks", "label": "L1 banks", "group": "Memory", "kind": "select", "options": [8, 16, 32, 64, 128, 256, 512]},
    {"path": "l1.bytes_per_bank_cycle", "label": "Bytes/bank/cycle", "group": "Memory", "kind": "select", "options": [4, 8, 16, 32, 64]},
    {"path": "l1.efficiency", "label": "L1 efficiency", "group": "Memory", "kind": "num", **_PCT},
    {"path": "l1.state_reserve", "label": "L1 share for resident state", "group": "Memory", "kind": "num", "min": 0, "max": 0.5, "step": 0.05},
    {"path": "l1.tile_fraction", "label": "L1 share for matmul/attention tiles", "group": "Memory", "kind": "num", "min": 0.1, "max": 0.7, "step": 0.05},
    {"path": "hbm.gbs", "label": "HBM peak bandwidth", "unit": "GB/s", "group": "Memory", "kind": "num", "min": 50, "max": 2000, "step": 10},
    {"path": "hbm.efficiency", "label": "HBM sustained efficiency", "group": "Memory", "kind": "num", **_PCT},
    {"path": "hbm.gib", "label": "HBM capacity", "unit": "GiB", "group": "Memory", "kind": "num", "min": 0.5, "max": 64, "step": 0.5},
    {"path": "memory.weight_path", "label": "Weight path", "group": "Memory", "kind": "select", "options": ["via_l1", "direct"],
     "help": "via_l1: weights cross L1 twice (explorer). direct: weights stream HBM to the array, an architecture decision."},
    {"path": "precision.weights", "label": "Weight format", "group": "Precision", "kind": "select", "options": ["captured", "uniform_int4", "uniform_int8"],
     "help": "captured = the real Q4_K_M mix. Q4_K_M does not validate a custom INT4 format."},
    {"path": "precision.act_bits", "label": "Activation bits", "group": "Precision", "kind": "select", "options": [8, 16, 32]},
    {"path": "precision.kv_bits", "label": "KV cache bits", "group": "Precision", "kind": "select", "options": [8, 16, 32]},
    {"path": "precision.state_bits", "label": "Recurrent state bits", "group": "Precision", "kind": "select", "options": [16, 32]},
    {"path": "attention.causal_skip", "label": "Skip masked attention tiles", "group": "Attention", "kind": "bool"},
    {"path": "attention.mask_onchip", "label": "Generate the mask on-chip", "group": "Attention", "kind": "bool"},
    {"path": "attention.kv_padding", "label": "KV length read", "group": "Attention", "kind": "select", "options": ["captured", "exact"]},
    {"path": "attention.overlap_softmax", "label": "Softmax overlaps matmul", "group": "Attention", "kind": "bool"},
    {"path": "attention.kv_tile", "label": "KV tile", "group": "Attention", "kind": "select", "options": [32, 64, 128, 256, 512]},
    {"path": "memory.residency", "label": "Activation traffic model", "group": "Modelling switches", "kind": "select", "options": ["belady", "per_op"],
     "help": "belady: liveness simulation. per_op: the explorer's spill proxy."},
    {"path": "memory.fusion", "label": "Fusion", "group": "Modelling switches", "kind": "select", "options": ["chains", "groups", "none"],
     "help": "chains: single-consumer elementwise epilogues. groups: anchored fusion groups from the dependency map (more optimistic). none: pessimistic."},
    {"path": "memory.l1_feed", "label": "L1 feed model", "group": "Modelling switches", "kind": "select", "options": ["array", "tile"]},
    {"path": "matrix.tile_mode", "label": "Tiling", "group": "Modelling switches", "kind": "select", "options": ["search", "fixed"]},
    {"path": "schedule.mode", "label": "Whole-graph time", "group": "Modelling switches", "kind": "select", "options": ["pools", "serial"]},
    {"path": "schedule.within_op_overlap", "label": "Overlap compute, L1 and HBM inside an op", "group": "Modelling switches", "kind": "bool"},
    {"path": "sync_cycles", "label": "Cross-unit hand-off", "unit": "cycles", "group": "Modelling switches", "kind": "int", "min": 0, "max": 2000, "step": 10},
    {"path": "launch_cycles", "label": "Launch overhead per op", "unit": "cycles", "group": "Modelling switches", "kind": "int", "min": 0, "max": 500, "step": 5},
    {"path": "host.enabled", "label": "Host fallback (measured CPU time)", "group": "Modelling switches", "kind": "bool",
     "help": "Ops no unit can run take their measured CPU time. Only fa-on graphs with a trace."},
    {"path": "resources.cost.l1_bank", "label": "Cost per L1 bank", "group": "Cost proxy (placeholders)", "kind": "num", "min": 0, "max": 64, "step": 1},
    {"path": "resources.cost.direct_weight_buffer_per_pe", "label": "Cost per PE, direct weight path", "group": "Cost proxy (placeholders)", "kind": "num", "min": 0, "max": 4, "step": 0.05},
    {"path": "resources.cost.matrix_pe", "label": "Cost per matrix PE", "group": "Cost proxy (placeholders)", "kind": "num", "min": 0, "max": 8, "step": 0.25},
    {"path": "resources.cost.vector_lane", "label": "Cost per vector lane", "group": "Cost proxy (placeholders)", "kind": "num", "min": 0, "max": 64, "step": 1},
    {"path": "resources.cost.recurrent_lane", "label": "Cost per recurrence lane", "group": "Cost proxy (placeholders)", "kind": "num", "min": 0, "max": 64, "step": 1},
    {"path": "resources.cost.sram_mib", "label": "Cost per MiB of L1", "group": "Cost proxy (placeholders)", "kind": "num", "min": 0, "max": 1024, "step": 8},
]
SWEEPABLE = {c["path"] for c in CONTROLS if c["kind"] in ("num", "int", "select", "bool")}


def _clean(obj):
    """JSON has no inf/nan: report them as null (e.g. tokens/s when a time is zero)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class App:
    def __init__(self, run_dir):
        self.run = Path(run_dir)
        self.names = sorted(d.name for d in (self.run / "graphs").glob("*")
                            if has_capture(d))
        self._workloads, self._lock = {}, threading.Lock()

    def workload(self, name):
        if name not in self.names:
            raise ApiError(f"unknown workload {name!r}")
        with self._lock:
            if name not in self._workloads:
                self._workloads[name] = load_workload(self.run / "graphs" / name)
            return self._workloads[name]

    @staticmethod
    def config(body, key="config"):
        try:
            return make_config(body.get(key) or {})
        except ValueError as e:
            raise ApiError(str(e))

    # -- routes ---------------------------------------------------------------------------------------------
    def meta(self):
        profiles = []
        for path in sorted(PROFILE_DIR.glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            profiles.append({"name": doc["name"], "description": doc.get("description", ""), "config": load_profile(path)[0]})
        workloads = []
        for name in self.names:
            w = self.workload(name)
            m = w.meta
            workloads.append({"name": name, "prompt_tokens": m["prompt_tokens"], "flash_attention": m.get("flash_attention"),
                              "context_capacity": m["context_capacity"], "nodes": len(w.prefill.nodes)})
        return {"engine_version": VERSION, "run": self.run.name, "workloads": workloads, "profiles": profiles,
                "controls": CONTROLS, "defaults": DEFAULTS}

    def eval(self, body):
        w, cfg = self.workload(body.get("graph")), self.config(body)
        top = min(int(body.get("top_nodes") or 0), 25)
        trace = load_trace(self.run, w) if cfg["host"]["enabled"] else None
        r = evaluate(w, cfg, trace, detail=bool(top))
        for p in r["phases"].values():
            rows = p.pop("rows", None)
            if rows:
                p["top_nodes"] = sorted(rows, key=lambda x: -x["duration_s"])[:top]
        r["summary"] = sweep.summarize(r)
        return r

    def sweep(self, body):
        w, base = self.workload(body.get("graph")), self.config(body)
        axes = body.get("axes") or {}
        if not axes or len(axes) > 2:
            raise ApiError("give one or two sweep axes")
        for path, values in axes.items():
            if path not in SWEEPABLE:
                raise ApiError(f"{path!r} is not a sweepable control")
            if not isinstance(values, list) or not values:
                raise ApiError(f"axis {path!r} needs a list of values")
        points = 1
        for values in axes.values():
            points *= len(values)
        if points > MAX_SWEEP_POINTS:
            raise ApiError(f"{points} points requested; the page limit is {MAX_SWEEP_POINTS}")
        trace = load_trace(self.run, w) if base["host"]["enabled"] else None
        records = sweep.run_sweep(w, base, axes, trace, limit=MAX_SWEEP_POINTS)
        front = sweep.pareto(records, (("cost", "min"), ("decode_s", "min")))
        return {"axes": axes, "records": records, "frontier": [records.index(r) for r in front]}

    def ablation(self, body):
        w, new = self.workload(body.get("graph")), self.config(body)
        legacy = load_profile("legacy-equations")[0]
        rows, loo = sweep.ablation(w, legacy, new)
        return {"cumulative": rows, "leave_one_out": loo}

    def sensitivity(self, body):
        w, cfg = self.workload(body.get("graph")), self.config(body)
        delta = float(body.get("delta") or 0.3)
        if not 0.05 <= delta <= 0.6:
            raise ApiError("delta must be between 0.05 and 0.6")
        base, rows = sweep.sensitivity(w, cfg, delta=delta)
        return {"base": base, "rows": rows, "delta": delta}


def make_handler(app):
    routes = {"/api/eval": app.eval, "/api/sweep": app.sweep, "/api/ablation": app.ablation, "/api/sensitivity": app.sensitivity}

    class Handler(BaseHTTPRequestHandler):
        server_version = "sbengine"

        def log_message(self, fmt, *args):     # quiet unless the server itself fails (4xx are the caller's mistakes)
            if args and str(args[1]).startswith("5"):
                super().log_message(fmt, *args)

        def _send(self, status, payload, content_type="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(_clean(payload), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/meta":
                self._send(200, app.meta())
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            fn = routes.get(self.path.split("?", 1)[0])
            if fn is None:
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    raise ApiError("request too large", 413)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ApiError("body must be a JSON object")
                self._send(200, fn(body))
            except ApiError as e:
                self._send(e.status, {"error": str(e)})
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": f"bad request: {e}"})

    return Handler


def serve(run_dir=None, host="127.0.0.1", port=8765, open_browser=False, results_dir=None):
    run = Path(run_dir) if run_dir else latest_run(results_dir)
    app = App(run)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"sbengine {VERSION} serving run {run.name} ({len(app.names)} workloads) at {url}   (Ctrl+C to stop)")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: bound to a non-loopback address; this server has no authentication")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
