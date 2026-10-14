"""Run corrected 2048-square reference cases in the Timeloop container."""
from pathlib import Path
import pytimeloop.timeloopfe.v4 as tl

ROOT = Path(__file__).resolve().parent
for name, mapping in (("decode", "bpu-8x8.map.yaml"),
                      ("prefill", "bpu-8x8-prefill.map.yaml")):
    spec = tl.Specification.from_yaml_files(
        str(ROOT / "arch" / "bpu-8x8.arch.yaml"),
        str(ROOT / "mapping" / mapping),
        str(ROOT / "problem" / f"qwen-{name}.prob.yaml"),
    )
    print(f"running corrected {name}", flush=True)
    tl.call_model(spec, output_dir=str(ROOT / "output" / "corrected-2048" / name))
