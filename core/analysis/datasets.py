"""Dataset role/build provenance for current and historical profiler runs."""
import json
from pathlib import Path


ROLE_LABELS = {
    "cpu_reference": "CPU reference",
    "cuda_build_cpu_control": "CPU control (CUDA build)",
    "cuda_build_cpu_no_host_control": "CPU control (CUDA build, no_host)",
    "accelerated_execution": "Accelerated execution",
    "unclassified_cpu": "CPU (build not identified)",
    "cpu_trace": "Instrumented CPU trace",
}


def _manifest_path(directory):
    directory = Path(directory).resolve()
    candidates = [directory]
    if directory.name == "analysis":
        candidates.append(directory.parent)
    if directory.parent.name == "prompts":
        candidates.append(directory.parent.parent)
    for candidate in candidates:
        path = candidate / "run-manifest.json"
        if path.is_file():
            return path
    # The immutable first experiment uses a package-level manifest with baseline_backends.
    path = directory.parent / "manifest.json"
    if directory.name == "results" and path.is_file():
        return path
    return None


def _cuda_enabled(build):
    if not build:
        return None
    requested = build.get("requested") or {}
    value = requested.get("GGML_CUDA")
    effective = (build.get("effective") or {}).get("cuda")
    if value is None:
        return effective
    enabled = str(value).upper() in ("1", "ON", "TRUE", "YES")
    return None if effective is not None and enabled != effective else enabled


def _inferred(manifest):
    names = list((manifest.get("config") or {}).get("backends") or [])
    legacy = manifest.get("baseline_backends")
    if legacy:
        names = [name.lower() for name in legacy]
    baseline_cuda = _cuda_enabled(manifest.get("build"))
    split_cuda = manifest.get("cuda_build") is not None
    rows = []
    for name in names:
        if name == "cpu":
            if baseline_cuda is True:
                role = "cuda_build_cpu_control"
            elif baseline_cuda is False or (legacy and "cuda" not in names):
                role = "cpu_reference"
            else:
                role = "unclassified_cpu"
            build = "legacy_baseline" if legacy else "baseline"
        elif name == "cuda-cpu":
            role, build = "cuda_build_cpu_control", "cuda"
        elif name == "cuda-cpu-no-host":
            role, build = "cuda_build_cpu_no_host_control", "cuda"
        elif name in ("cuda", "metal"):
            role, build = "accelerated_execution", "cuda" if name == "cuda" and split_cuda else "baseline"
        else:
            role, build = "unknown", "unknown"
        rows.append({"name": name, "role": role, "build": build, "provenance": "inferred"})
    return rows


def read(directory):
    """Return dataset records. Historical manifests are classified only when their build identity proves the role."""
    path = _manifest_path(directory)
    if path is None:
        directory = Path(directory)
        names = []
        for candidate in ("cuda-cpu-no-host", "cuda-cpu", "cuda", "metal", "cpu"):
            if any((directory / f"{candidate}-{tag}{suffix}").is_file()
                   for tag in ("baseline", "8k") for suffix in (".jsonl", ".jsonl.gz")):
                names.append(candidate)
        rows = []
        for name in names:
            if name == "cpu":
                role, build = "unclassified_cpu", "unknown"
            elif name == "cuda-cpu":
                role, build = "cuda_build_cpu_control", "cuda"
            elif name == "cuda-cpu-no-host":
                role, build = "cuda_build_cpu_no_host_control", "cuda"
            else:
                role, build = "accelerated_execution", "unknown"
            rows.append({"name": name, "role": role, "build": build, "provenance": "filenames_only"})
        return rows
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("datasets")
    if rows is None:
        rows = _inferred(manifest)
    names = [row.get("name") for row in rows]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError(f"Invalid dataset provenance in {path}")
    for row in rows:
        build_key = {"baseline": "build", "cuda": "cuda_build", "profile": "profile_build"}.get(row.get("build"))
        row["build_config"] = manifest.get(build_key) if build_key else None
        if row.get("role") == "cpu_reference" and row.get("build") != "legacy_baseline":
            cuda = _cuda_enabled(row["build_config"])
            if cuda is True or row.get("build") != "baseline":
                raise ValueError(f"CPU-reference role contradicts its CUDA-enabled build: {path}")
            if cuda is None:
                row["role"] = "unclassified_cpu"
    return rows


def for_run(rows, run_name):
    matches = [row for row in rows if run_name == row["name"] or run_name.startswith(row["name"] + "-")]
    if not matches:
        return None
    row = max(matches, key=lambda row: len(row["name"]))
    suffix = run_name[len(row["name"]):]
    if "profile" in suffix:
        return {"name": run_name, "role": "cpu_trace", "build": "profile"}
    if "control" in suffix:
        return {"name": run_name, "role": "unknown", "build": "unknown"}
    return row


def cpu_reference(directory):
    matches = [row for row in read(directory) if row.get("role") == "cpu_reference"]
    if len(matches) != 1:
        raise ValueError("Run does not identify exactly one CPU-reference dataset; refusing to guess from filenames")
    row = matches[0]
    if row.get("build") != "legacy_baseline" and _cuda_enabled(row.get("build_config")) is not False:
        raise ValueError("CPU reference has no verified CPU build identity")
    return row


def label(row):
    if row is None:
        return "Unclassified"
    if row.get("role") == "accelerated_execution":
        return row["name"].upper() if row["name"] == "cuda" else row["name"].title()
    return ROLE_LABELS.get(row.get("role"), row.get("role", "Unknown").replace("_", " ").title())
