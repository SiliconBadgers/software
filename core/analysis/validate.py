"""Read-only comparison of saved checkpoints, not sequence/task quality."""
import argparse
import json

import numpy as np

from policy import case_prefix, is_diagnostic, load_experiment, profile_prefixes, DEFAULT_EXPERIMENT
from result_io import add_paths, exists, open_result, output_dir, write_json


def read_logits(path, vocabulary):
    with open_result(path, "rb") as stream:
        values = np.frombuffer(stream.read(), dtype="<f4")
    if values.shape != (vocabulary,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid saved logits: {path}")
    return values


def validate(results_dir, experiment, lengths, skip_metal, require_profile=True):
    vocabulary = experiment["model"]["vocabulary_size"]
    profile_checks, backend_checks, missing_profile = [], [], []
    for n in lengths:
        cpu_prefix, metal_prefix = case_prefix(experiment, "cpu", n), case_prefix(experiment, "metal", n)
        for phase in ("prefill", "decode"):
            suffix = f"-p{n}-{phase}.f32"
            cpu = read_logits(results_dir / (cpu_prefix + suffix), vocabulary)
            if not skip_metal:
                metal = read_logits(results_dir / (metal_prefix + suffix), vocabulary)
                backend_checks.append(dict(
                    prompt_tokens=n, phase=phase, elements=cpu.size, both_finite=True,
                    max_abs_difference=float(np.max(np.abs(cpu - metal))),
                    mean_abs_difference=float(np.mean(np.abs(cpu - metal))),
                    cpu_top_token=int(cpu.argmax()), metal_top_token=int(metal.argmax()),
                    top10_overlap=len(set(cpu.argsort()[-10:]) & set(metal.argsort()[-10:]))))
            found = False
            for prefix in (profile_prefixes(experiment, n) if require_profile else []):
                path = results_dir / (prefix + suffix)
                if not exists(path):
                    continue
                found = True
                prof = read_logits(path, vocabulary)
                same = cpu.tobytes() == prof.tobytes()
                profile_checks.append(dict(run=prefix, prompt_tokens=n, phase=phase, bit_identical=same,
                                           max_abs_difference=float(np.max(np.abs(cpu - prof)))))
                if not same:
                    raise ValueError(f"Profiler output differs: {prefix} {n} {phase}")
            if not found and require_profile:
                if not is_diagnostic(experiment, n):
                    raise FileNotFoundError(f"Missing required CPU profile checkpoint: {n} {phase}")
                missing_profile.append(dict(prompt_tokens=n, phase=phase))
    return profile_checks, backend_checks, missing_profile


def write_outputs(out_dir, profile_checks, backend_checks):
    write_json(out_dir / "profile-output-validation.json", profile_checks)
    write_json(out_dir / "cpu-metal-logit-check.json", backend_checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument("--lengths", type=int, nargs="+", help="Default: the experiment's prompt lengths")
    parser.add_argument("--skip-metal", action="store_true", help="Validate CPU profiler outputs only")
    args = parser.parse_args(argv)
    experiment = load_experiment(args.experiment)
    results_dir = args.results_dir.resolve()
    out = output_dir(args.output_dir, default_temp=False)
    lengths = args.lengths or experiment["workload"]["prompt_lengths"]
    skip_metal = args.skip_metal
    if not skip_metal and not exists(results_dir / "metal-baseline.jsonl"):
        print("Note: no Metal baseline in this run; backend comparison skipped.")
        skip_metal = True
    profile_checks, backend_checks, missing = validate(results_dir, experiment, lengths, skip_metal)
    if out is not None:
        write_outputs(out, profile_checks, backend_checks)
    print(json.dumps(dict(profile_checks=profile_checks, backend_checks=backend_checks,
                          missing_optional_diagnostic_profile=missing), indent=2))
    if out is not None:
        print("Output directory:", out)


if __name__ == "__main__":
    main()
