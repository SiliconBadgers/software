"""Run profiled Qwen FFN cases with the RTL-aligned A-stationary mapping."""
from pathlib import Path
import sys
import pytimeloop.timeloopfe.v4 as tl

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "qwen-matmuls"
CASES = [
    ("ffn-gate-decode", "ffn-gate-decode.prob.yaml"),
    ("ffn-gate-prefill", "ffn-gate-prefill.prob.yaml"),
    ("ffn-out-decode", "ffn-out-decode.prob.yaml"),
    ("ffn-out-prefill", "ffn-out-prefill.prob.yaml"),
    ("output-head", "output-head.prob.yaml"),
]

for stem, problem in CASES:
    if len(sys.argv) > 1 and stem not in sys.argv[1:]:
        continue
    families = ["a-stationary"]
    if "prefill" in stem:
        families.append("weight-reuse")
    for family in families:
        arch = ROOT / "arch" / "bpu-8x8.arch.yaml"
        mapping = ROOT / "mapping" / f"{stem}-{family}.map.yaml"
        prob = ROOT / "problem" / problem
        out = OUT / f"{stem}-{family}"
        print(f"running {stem} / {family}", flush=True)
        spec = tl.Specification.from_yaml_files(str(arch), str(mapping), str(prob))
        tl.call_model(spec, output_dir=str(out))
