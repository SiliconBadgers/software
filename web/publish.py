#!/usr/bin/env python3
"""Publish one build while retaining main and other branch previews."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def preview_name(ref):
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", ref).strip("-").lower()[:55] or "branch"
    return slug + "-" + hashlib.sha256(ref.encode()).hexdigest()[:8]


def overlay(site, build, ref):
    if ref == "main":
        # Preview editions survive replacement of the main edition.
        for item in site.iterdir():
            if item.name in {".git", "previews"}: continue
            if item.is_dir(): shutil.rmtree(item)
            else: item.unlink()
        destination = site
    else:
        destination = site / "previews" / preview_name(ref)
        if destination.exists(): shutil.rmtree(destination)
    shutil.copytree(build, destination, dirs_exist_ok=True)
    write_editions(site)
    return destination


def write_editions(site):
    """Index every published revision from its own build metadata."""
    editions = []
    folders = [site]
    preview_root = site / "previews"
    if preview_root.is_dir():
        folders.extend(path for path in sorted(preview_root.iterdir())
                       if path.is_dir() and not path.is_symlink()
                       and re.fullmatch(r"[a-z0-9-]+", path.name))
    for folder in folders:
        metadata = folder / "build-info.json"
        if not metadata.is_file():
            continue
        info = json.loads(metadata.read_text())
        if (not isinstance(info.get("ref"), str) or not info["ref"]
                or not re.fullmatch(r"[a-f0-9]{40}", str(info.get("sha", "")))):
            raise ValueError(f"Invalid edition metadata: {metadata}")
        editions.append({"ref": info["ref"], "sha": info["sha"],
                         "preview": folder != site,
                         "path": "./" if folder == site else f"previews/{folder.name}/"})
    (site / "editions.json").write_text(
        json.dumps({"schemaVersion": 1, "editions": editions}, indent=2) + "\n")


def run(*args, cwd=None):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        # Git's diagnostics explain authentication and branch-policy failures.
        print(result.stderr, file=sys.stderr, end="")
        result.check_returncode()
    return result


def authentication_environment(header, url="https://github.com/"):
    key = f"http.{url}.extraheader"
    # extraheader is multi-valued: reset the checkout's value before forwarding it. Disable credential helpers
    # for this non-interactive operation so a rejected header fails instead of opening a desktop prompt.
    return {"GIT_CONFIG_COUNT": "3", "GIT_CONFIG_KEY_0": key, "GIT_CONFIG_VALUE_0": "",
            "GIT_CONFIG_KEY_1": key, "GIT_CONFIG_VALUE_1": header,
            "GIT_CONFIG_KEY_2": "credential.helper", "GIT_CONFIG_VALUE_2": ""}


def publish(build, ref):
    remote = run("git", "remote", "get-url", "origin").stdout.strip()
    header = subprocess.run(["git", "config", "--get", "http.https://github.com/.extraheader"],
                            capture_output=True, text=True)
    if header.returncode == 0:
        # Forward the checkout token without saving it in the publication branch.
        os.environ.update(authentication_environment(header.stdout.strip()))
    with tempfile.TemporaryDirectory(prefix="sb-software-pages-") as temp:
        site = Path(temp) / "site"
        exists = run("git", "ls-remote", "--heads", "origin", "gh-pages").stdout.strip()
        if exists:
            run("git", "clone", "--depth", "1", "--branch", "gh-pages", remote, str(site))
        else:
            site.mkdir()
            run("git", "init", "-b", "gh-pages", cwd=site)
            run("git", "remote", "add", "origin", remote, cwd=site)
        run("git", "config", "user.name", "github-actions[bot]", cwd=site)
        run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com", cwd=site)
        destination = overlay(site, build, ref)
        run("git", "add", "--all", cwd=site)
        changed = run("git", "diff", "--cached", "--name-only", cwd=site).stdout
        if changed:
            run("git", "commit", "-m", f"Publish software workspace: {ref}", cwd=site)
            # A normal push retains deployment history; workflows serialize publication.
            run("git", "push", "origin", "HEAD:gh-pages", cwd=site)
        print("/" if ref == "main" else f"/previews/{destination.name}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()
    publish(args.build.resolve(), args.ref)
