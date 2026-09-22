"""Read-only comparison of saved checkpoints, not sequence/task quality."""
import argparse
import json
import numpy as np
from result_io import add_paths, output_dir, exists, open_result

parser = argparse.ArgumentParser(description=__doc__)
add_paths(parser)
parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
parser.add_argument("--skip-metal", action="store_true", help="Validate CPU profiler outputs only")
args = parser.parse_args()
R = args.results_dir.resolve()
O = output_dir(args.output_dir, default_temp=False)
profile_checks, backend_checks, missing_profile = [], [], []


def read_logits(path):
    with open_result(path, "rb") as stream:
        values = np.frombuffer(stream.read(), dtype="<f4")
    if values.shape != (248320,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid saved logits: {path}")
    return values


for n in args.lengths:
    cpu_prefix = "cpu-8k" if n == 8192 else "cpu-baseline"
    metal_prefix = "metal-8k" if n == 8192 else "metal-baseline"
    profile_prefixes = ["cpu-8k-profile", "cpu-8k-profile-repeat"] if n == 8192 else ["cpu-profile"]
    for phase in ("prefill", "decode"):
        suffix = f"-p{n}-{phase}.f32"
        cpu = read_logits(R / (cpu_prefix + suffix))
        if not args.skip_metal:
            metal = read_logits(R / (metal_prefix + suffix))
            backend_checks.append(dict(prompt_tokens=n, phase=phase, elements=cpu.size,
                both_finite=True, max_abs_difference=float(np.max(np.abs(cpu-metal))),
                mean_abs_difference=float(np.mean(np.abs(cpu-metal))),
                cpu_top_token=int(cpu.argmax()), metal_top_token=int(metal.argmax()),
                top10_overlap=len(set(cpu.argsort()[-10:]) & set(metal.argsort()[-10:]))))
        found = False
        for prefix in profile_prefixes:
            path = R / (prefix + suffix)
            if not exists(path):
                continue
            found = True
            prof = read_logits(path)
            same = cpu.tobytes() == prof.tobytes()
            profile_checks.append(dict(run=prefix, prompt_tokens=n, phase=phase,
                bit_identical=same, max_abs_difference=float(np.max(np.abs(cpu-prof)))))
            if not same:
                raise ValueError(f"Profiler output differs: {prefix} {n} {phase}")
        if not found:
            if n != 8192:
                raise FileNotFoundError(f"Missing required CPU profile checkpoint: {n} {phase}")
            missing_profile.append(dict(prompt_tokens=n, phase=phase))
if O is not None:
    (O / "profile-output-validation.json").write_text(json.dumps(profile_checks, indent=2))
    (O / "cpu-metal-logit-check.json").write_text(json.dumps(backend_checks, indent=2))
print(json.dumps(dict(profile_checks=profile_checks, backend_checks=backend_checks,
                     missing_optional_8k_profile=missing_profile), indent=2))
if O is not None:
    print("Output directory:", O)
