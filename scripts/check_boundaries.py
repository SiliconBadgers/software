"""Standard-library checks for scientific stage and evidence-reference boundaries."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from sbengine.workload import load_workload
from sbengine.graph import Graph
from sbengine.provenance import validate_capture, validate_captures
from sweeps.specs import load_spec, reference

HISTORICAL = ("experiments/", "research/", "results/")
REGRESSION_STUDIES = ("compute/capacity.json", "local-memory/service.json", "joint/matrix-l1.json")
COMPRESSION_SUFFIXES = {".gz", ".gzip", ".bz2", ".xz", ".lzma", ".zst", ".zstd", ".lz4", ".br",
                        ".zip", ".7z", ".rar", ".tar", ".tgz", ".tbz2", ".txz"}


def tracked_files(root):
    raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    files = {}
    for entry in filter(None, raw.decode().split("\0")):
        info, path = entry.split("\t", 1)
        mode, blob, stage = info.split()
        if stage != "0":
            raise ValueError(f"unmerged tracked file: {path}")
        files[path] = {"mode": mode, "git_blob": blob}
    return files


def historical_files(root=ROOT, tracked=None):
    """Check the frozen subset against the index/worktree, without old Git objects."""
    fixtures = root / "engine/tests/fixtures"
    history = json.loads((fixtures / "historical-0e06023.json").read_text(encoding="utf-8"))
    fixture = json.loads((fixtures / "characterization-0e06023.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(history, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if {"count": len(history), "sha256": digest} != fixture["historical_files"]:
        raise ValueError("historical inventory differs from the original characterization digest")
    gitlinks = fixture["historical_gitlinks"]
    expected = {**history, **{path: {"mode": "160000", "git_blob": blob} for path, blob in gitlinks.items()}}
    tracked = tracked_files(root) if tracked is None else tracked
    for path, identity in expected.items():
        if tracked.get(path) != identity:
            raise ValueError(f"preserved historical identity changed or missing: {path}")
    # Porcelain diff refreshes stat-only changes before comparing content.
    dirty = subprocess.check_output(
        ["git", "diff", "--no-ext-diff", "--no-textconv", "--name-only", "-z",
         "--ignore-submodules=dirty", "--", *HISTORICAL], cwd=root)
    for path in filter(None, dirty.decode().split("\0")):
        if path in expected:
            raise ValueError(f"preserved historical working tree changed: {path}")
    # Hash content even when cached size/timestamps match. Accept exact blobs or
    # Git-cleaned checkout bytes, retaining historical CRLF blobs as recorded.
    for path in history:
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"preserved historical working tree changed: {path}")
    blobs = subprocess.check_output(
        ["git", "hash-object", "--stdin-paths"], cwd=root,
        input="".join(json.dumps(path, ensure_ascii=False) + "\n" for path in history).encode()).decode().splitlines()
    if len(blobs) != len(history):
        raise ValueError("historical working-tree content comparison is incomplete")
    mismatched = [path for path, blob in zip(history, blobs) if blob != history[path]["git_blob"]]
    if mismatched:
        raw_blobs = subprocess.check_output(
            ["git", "hash-object", "--stdin-paths", "--no-filters"], cwd=root,
            input="".join(json.dumps(path, ensure_ascii=False) + "\n" for path in mismatched).encode()).decode().splitlines()
        if len(raw_blobs) != len(mismatched):
            raise ValueError("historical working-tree content comparison is incomplete")
        for path, blob in zip(mismatched, raw_blobs):
            if blob != history[path]["git_blob"]:
                raise ValueError(f"preserved historical working tree changed: {path}")
    return history, gitlinks


def study_paths(root=ROOT):
    """Retain the original regression cases without closing the catalog to additions."""
    directory = root / "engine/sweeps"
    paths = sorted(directory.glob("*/*.json"))
    present = {path.relative_to(directory).as_posix() for path in paths}
    missing = sorted(set(REGRESSION_STUDIES) - present)
    if missing:
        raise ValueError("missing regression studies: " + ", ".join(missing))
    return paths


def import_errors(path, source):
    errors = []
    path = str(path).replace("\\", "/")
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = ([node.module] + [f"{node.module}.{item.name}" for item in node.names]
                     if node.module else [item.name for item in node.names])
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
            if path.startswith("engine/sbengine/codegen/") and any(
                    part in {"costs", "model", "sweep", "web", "cli", "cpu_validate", "trace"}
                    for part in name.split(".")):
                errors.append(f"{path}:{node.lineno}: codegen imports evaluation/presentation {name}")
    return errors


def tracked_path_error(path, preserved=False):
    # Git uses slash-separated paths. Account for case and Win32 trailing-dot/
    # space aliases, without banning unrelated research or prose extensions.
    parts = tuple(part.casefold().rstrip(" .") for part in PurePosixPath(path.replace("\\", "/")).parts)
    if not parts:
        return None
    if parts[0] in {"build", "work", ".venv"}:
        return f"generated/local path tracked: {path}"
    if preserved:
        return None
    name = parts[-1].split(":", 1)[0].rstrip(" .")  # NTFS stream spelling does not change the base family.
    while PurePosixPath(name).suffix in COMPRESSION_SUFFIXES:
        name = name[:-len(PurePosixPath(name).suffix)].rstrip(" .")
    if name.endswith((".gguf", ".safetensors", ".f32", ".jsonl")) or name in {
            "records.json", "timeline.json", "pool-schedule.json"}:
        return f"new bulk output tracked: {path}"
    return None


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
                validate_capture(Graph(captured, capture.rsplit("/", 1)[1]), capture, record)
            for workload in sorted({capture.rsplit("/", 1)[0] for capture in record["captures"]}):
                loaded = load_workload(root / record["run"] / "graphs" / workload)
                validate_captures(loaded, record)
        except (OSError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    try:
        specs = study_paths(root)
    except ValueError as error:
        errors.append(str(error))
        specs = []
    for path in specs:
        try:
            spec, baseline, _ = load_spec(root, path)
            reference(root, f"{baseline['run']}/graphs/{spec['workload']}/prefill.json.gz")
            loaded = load_workload(root / baseline["run"] / "graphs" / spec["workload"])
            validate_captures(loaded, baseline)
        except (OSError, KeyError, TypeError, ValueError) as error:
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
        for path in sorted((root / folder).rglob("*.py")):
            errors.extend(import_errors(path.relative_to(root), path.read_text(encoding="utf-8")))
    tracked = tracked_files(root)
    preserved = set()
    try:
        history, gitlinks = historical_files(root, tracked)
        preserved = set(history) | set(gitlinks)
    except (OSError, ValueError) as error:
        errors.append(str(error))
    for path in tracked:
        error = tracked_path_error(path, path in preserved)
        if error:
            errors.append(error)
    errors.extend(reference_errors(root))
    return errors


if __name__ == "__main__":
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        raise SystemExit(1)
    print("PASS scientific imports, baseline/spec references and tracked artifact boundaries")
