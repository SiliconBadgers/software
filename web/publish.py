#!/usr/bin/env python3
"""Publish one build while retaining main and other branch previews."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
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
    return destination


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, text=True, check=True, capture_output=True)


def publish(build, ref):
    remote = run("git", "remote", "get-url", "origin").stdout.strip()
    header = subprocess.run(["git", "config", "--get", "http.https://github.com/.extraheader"],
                            capture_output=True, text=True)
    if header.returncode == 0:
        # Forward the checkout token without saving it in the publication branch.
        os.environ.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                          GIT_CONFIG_VALUE_0=header.stdout.strip())
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
