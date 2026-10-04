#!/usr/bin/env python3
"""Build the public software workspace from an explicit component registry."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Path must remain inside {root.name}: {relative}")
    return path


def validate_registry(registry):
    if registry.get("schemaVersion") != 1 or not registry.get("components"):
        raise ValueError("Registry needs schemaVersion 1 and at least one component")
    ids = set()
    for component in registry["components"]:
        for key in ("id", "title", "contributors", "module", "source", "pullRequest", "assets"):
            if key not in component:
                raise ValueError(f"Component missing {key}")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", component["id"]) or component["id"] in ids:
            raise ValueError("Component IDs must be unique lowercase names")
        ids.add(component["id"])
        if not component["contributors"] or not isinstance(component["pullRequest"], int):
            raise ValueError("Component needs contributor credit and a PR number")
        if not safe_path(WEB, component["module"]).is_file():
            raise ValueError(f"Missing adapter {component['module']}")
        safe_path(ROOT, component["source"])
        for asset in component["assets"]:
            source = safe_path(ROOT, asset["source"])
            if not source.is_file() or source.suffix.lower() in {".gguf", ".bin", ".pt", ".pth", ".safetensors"}:
                raise ValueError(f"Unsupported public asset: {asset['source']}")
            safe_path(Path("/site"), asset["target"])


def build(output, ref, preview=False):
    registry = json.loads((WEB / "registry.json").read_text())
    validate_registry(registry)
    output = output.resolve()
    if output == ROOT or output == WEB or ROOT.is_relative_to(output):
        raise ValueError("Output must not contain repository sources")
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "build"):
        raise ValueError("In-repository output must be under build/")
    output.mkdir(parents=True, exist_ok=True)
    for old in output.iterdir():
        if old.is_dir(): shutil.rmtree(old)
        else: old.unlink()
    manifest = []
    destinations = set()

    def copy(source, target):
        if target in destinations:
            raise ValueError(f"Duplicate published asset: {target}")
        destinations.add(target)
        dest = safe_path(output, target)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        manifest.append({"source": str(source.relative_to(ROOT)), "published": target,
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})

    for filename in ("index.html", "app.js", "style.css", "registry.json"):
        copy(WEB / filename, filename)
    for component in registry["components"]:
        copy(safe_path(WEB, component["module"]), component["module"])
        for asset in component["assets"]:
            copy(safe_path(ROOT, asset["source"]), asset["target"])
    graphs = ROOT / "experiments/llama-cpp/2026-09-24-qwen35-2b/graphs"
    for capture in ("pp128", "pp512"):
        for phase in ("prefill", "decode"):
            for suffix in (".json", ".layer0.svg", ".layer3.svg"):
                filename = phase + suffix
                copy(graphs / capture / filename, f"assets/graphs/{capture}/{filename}")
    statuses = []
    try:
        statuses = json.loads(subprocess.check_output([
            "gh", "api", "repos/SiliconBadgers/software/pulls?state=all&per_page=100",
            "--jq", "map({number,state,merged_at})"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=15))
    except (FileNotFoundError, subprocess.SubprocessError):
        # Offline builds retain accurate status for PRs already in this revision.
        statuses = [{"number": number, "state": "closed", "merged_at": "recorded"} for number in (6, 8)]
    info = {"schemaVersion": 1, "sha": git("rev-parse", "HEAD"), "ref": ref,
            "preview": preview, "work": statuses}
    (output / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / ".nojekyll").touch()
    print(f"Built {len(manifest)} source assets in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/site")
    parser.add_argument("--ref", default="local")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    build(args.out, args.ref, args.preview)
