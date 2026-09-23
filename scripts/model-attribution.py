#!/usr/bin/env python3
"""Render reported model IDs from Git AI notes and commit declarations.

Read-only, standard library only. No prompts, session IDs or author emails are
published. Missing notes are not evidence of human authorship.
"""
import argparse
import html
import json
from pathlib import Path
import subprocess


def git(*args, optional=False):
    result = subprocess.run(["git", *args], text=True, capture_output=True, timeout=30)
    if result.returncode and not optional:
        raise RuntimeError(result.stderr.strip())
    return None if result.returncode else result.stdout.strip()


def recorded_models(note):
    if not note:
        return [], "No Git AI note"
    try:
        _, metadata = note.split("\n---\n", 1)
        data = json.loads(metadata)
        schema = data.get("schema_version", "")
        if schema.startswith("authorship/3."):
            records = data.get("sessions", {})
        elif schema.startswith("authorship/2."):
            records = data.get("prompts", {})
        else:
            return [], "Unsupported notes schema; inspect locally"
        pairs = set()
        incomplete = False
        for record in records.values():
            agent = record.get("agent_id", {})
            tool, model = agent.get("tool"), agent.get("model")
            if not isinstance(tool, str) or not tool.strip():
                tool = "unknown-tool"
            if not isinstance(model, str) or not model.strip():
                model = "unknown-model"
            if tool == "unknown-tool" or model.lower() in ("unknown-model", "unknown", "auto", "default"):
                incomplete = True
            pairs.add((tool, model))
        return sorted(pairs), "Model ID missing/unresolved" if incomplete else ("Captured metadata" if pairs else "No model metadata")
    except (ValueError, TypeError, AttributeError):
        return [], "Unreadable note; inspect locally"


def declared_models(message):
    # Git itself parses trailer placement and continuation syntax.
    result = subprocess.run(["git", "interpret-trailers", "--parse"], input=message,
                            text=True, capture_output=True, check=True, timeout=10)
    values = []
    for line in result.stdout.splitlines():
        key, _, value = line.partition(":")
        if key.lower() == "ai-model":
            values.append(value.strip())
    return sorted(set(values))


def cell(text):
    # Notes/commit messages are untrusted. Keep them inert in Markdown/HTML.
    text = html.escape(str(text), quote=True)
    for char in "|`[]*_\\":
        text = text.replace(char, f"&#{ord(char)};")
    return text.replace("\n", " ").replace("\r", " ")


def render(rows):
    out = ["# Model attribution", "",
           "Exact IDs as reported by tools or declared by contributors. A provider may expose an alias rather than a backend snapshot.", "",
           "| Commit | Models recorded in Git AI notes | Declared AI-Model trailers | Visibility |",
           "|---|---|---|---|"]
    for row in rows:
        models = "; ".join(f"{tool}={model}" for tool, model in row["models"])
        declared = "; ".join(row["declared"])
        status = row["status"]
        if row["merge"] and not models and not declared:
            status = "Merge without own metadata; inspect contributing commits"
        elif models and not declared:
            status += "; missing model declaration"
        elif row["declared"] == ["none"] and models:
            status += "; CONFLICT: declared none but AI recorded"
        elif not models and row["declared"] == ["none"]:
            status += "; contributor declares human-only"
        if models and declared:
            missing = {f"{tool}={model}" for tool, model in row["models"]} - set(row["declared"])
            if missing:
                status += "; recorded IDs omitted from declaration"
        out.append(f"| {cell(row['commit'][:12])} | {cell(models or 'Unavailable')} | {cell(declared or 'Not declared')} | {cell(status)} |")
    if not rows:
        out.append("| No commits in range | | | |")
    out += ["", "Captured notes can retain attribution from earlier edits. These rows identify recorded models, not per-PR line totals or model cost.",
            "Declarations can also describe research/review or pasted output that an editing hook did not capture. Review unexplained differences.",
            "Missing metadata does not prove human authorship. This report is informational; it cannot detect undisclosed AI use or resolve hidden model routing.",
            "Prompts, transcripts, session identifiers and author emails are excluded.", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Exclusive base commit; omit to inspect just --head")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    head = git("rev-parse", "--verify", f"{args.head}^{{commit}}")
    if args.base and set(args.base) != {"0"}:
        base = git("rev-parse", "--verify", f"{args.base}^{{commit}}")
        commits = git("rev-list", "--reverse", f"{base}..{head}").splitlines()
    else:
        commits = [head]
    rows = []
    for commit in commits:
        note = git("notes", "--ref=ai", "show", commit, optional=True)
        models, status = recorded_models(note)
        rows.append({"commit": commit, "models": models, "status": status,
                     "declared": declared_models(git("show", "-s", "--format=%B", commit)),
                     "merge": len(git("show", "-s", "--format=%P", commit).split()) > 1})
    report = render(rows)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
