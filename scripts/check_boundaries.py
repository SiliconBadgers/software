"""Standard-library checks for scientific stage and evidence-reference boundaries."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ("experiments/", "research/", "results/")


def import_errors(path, source):
    errors = []
    path = str(path).replace("\\", "/")
    for node in ast.walk(ast.parse(source)):
        names = ([item.name for item in node.names] if isinstance(node, ast.Import) else
                 [node.module] if isinstance(node, ast.ImportFrom) and node.module else
                 [item.name for item in node.names] if isinstance(node, ast.ImportFrom) else [])
        for name in names:
            root = name.split(".")[0]
            if path.startswith("core/cdfg/"):
                local = isinstance(node, ast.ImportFrom) and node.level == 1
                if not local and root not in sys.stdlib_module_names and root != "cdfg":
                    errors.append(f"{path}:{node.lineno}: CDFG imports non-core dependency {name}")
                if isinstance(node, ast.ImportFrom) and node.level > 1:
                    errors.append(f"{path}:{node.lineno}: CDFG import escapes its package")
            if path == "engine/sbengine/mapping.py" and any(
                    part in {"costs", "model", "sweep", "web", "cli", "cpu_validate", "trace"} for part in name.split(".")):
                errors.append(f"{path}:{node.lineno}: mapping imports later stage {name}")
            if path in {"engine/sbengine/memory.py", "engine/sbengine/schedule.py"} and any(
                    part in {"model", "sweep", "web", "cli"} for part in name.split(".")):
                errors.append(f"{path}:{node.lineno}: simulation imports later stage {name}")
    return errors


def tracked_path_error(path, preserved=False):
    parts = PurePosixPath(path).parts
    if not parts:
        return None
    if parts[0] in {"build", "work", ".venv"}:
        return f"generated/local path tracked: {path}"
    if preserved:
        return None
    if path.endswith((".gguf", ".safetensors", ".f32", ".f32.gz", ".jsonl", ".jsonl.gz")) or parts[-1] in {
            "records.json", "timeline.json", "pool-schedule.json"}:
        return f"new bulk output tracked: {path}"
    return None


def reference(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaping repository reference: {name}")
    return path


def reference_errors(root):
    errors = []
    for path in sorted((root / "engine/designs/baselines").glob("*/record.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            profile = json.loads(reference(root, record["profile"]).read_text(encoding="utf-8"))
            experiment = json.loads(reference(root, record["experiment"]).read_text(encoding="utf-8"))
            if not profile.get("resolved"):
                raise ValueError("baseline profile is not resolved")
            if record["model"] != experiment["model"] or record["llama_cpp"] != experiment["llama_cpp"]:
                raise ValueError("baseline changes pinned model/source identity")
            if record["parent"]:
                reference(root, record["parent"])
            reference(root, record["decisions"])
            for evidence in record["evidence"]:
                reference(root, evidence)
            for capture, values in record["captures"].items():
                captured = reference(root, f"{record['run']}/graphs/{capture}.json.gz")
                if hashlib.sha256(captured.read_bytes()).hexdigest() != values[0]:
                    raise ValueError(f"capture identity differs: {capture}")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    for path in sorted((root / "engine/sweeps").glob("*/*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
            baseline = json.loads(reference(root, spec["baseline"]).read_text(encoding="utf-8"))
            profile = json.loads(reference(root, baseline["profile"]).read_text(encoding="utf-8"))["config"]
            reference(root, spec["reference"])
            reference(root, f"{baseline['run']}/graphs/{spec['workload']}/prefill.json.gz")
            decisions = json.loads(reference(root, baseline["decisions"]).read_text(encoding="utf-8"))
            known = {item["id"] for item in decisions["branches"]}
            if spec["decision"] not in known or not set(spec["prerequisites"]) <= known:
                raise ValueError("unknown decision/prerequisite")
            for key in spec["axes"]:
                value = profile
                for part in key.split("."):
                    value = value[part]
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    registry = json.loads((root / "web/registry.json").read_text(encoding="utf-8"))
    for component in registry["components"]:
        for asset in component["assets"]:
            virtual = asset.get("virtual")
            if virtual and (PurePosixPath(virtual).is_absolute() or ".." in PurePosixPath(virtual).parts
                            or virtual.split("/")[0] not in {"core", "engine"}):
                errors.append(f"unsafe Python virtual path: {virtual}")
    return errors


def check(root=ROOT):
    errors = []
    for folder in ("core/cdfg", "engine/sbengine"):
        for path in sorted((root / folder).glob("*.py")):
            errors.extend(import_errors(path.relative_to(root), path.read_text(encoding="utf-8")))
    tracked = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root).decode().split("\0")
    history = {}
    for entry in filter(None, tracked):
        info, path = entry.split("\t", 1)
        mode, blob, stage = info.split()
        preserved = path.startswith(HISTORICAL)
        if preserved and mode != "160000":
            history[path] = {"mode": mode, "git_blob": blob}
        error = tracked_path_error(path, preserved)
        if error:
            errors.append(error)
    fixture = json.loads((root / "engine/tests/fixtures/characterization-0e06023.json").read_text())
    digest = hashlib.sha256(json.dumps(history, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != fixture["historical_files"]["sha256"]:
        errors.append("preserved historical input inventory changed; new bulk is not implicitly allowed")
    errors.extend(reference_errors(root))
    return errors


if __name__ == "__main__":
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        raise SystemExit(1)
    print("PASS scientific imports, baseline/spec references and tracked artifact boundaries")
