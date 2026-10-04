"""Portable readers, writers and output guards for profiling evidence."""
import argparse
import gzip
import json
from pathlib import Path
import tempfile

from policy import DEFAULT_EXPERIMENT, REPO, load_experiment

# Recorded experiments are immutable; core tools never write anywhere under experiments/.
PROTECTED = REPO / "experiments"


def result_path(path):
    path = Path(path)
    if path.is_file():
        return path
    compressed = Path(str(path) + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(path)


def exists(path):
    try:
        result_path(path)
        return True
    except FileNotFoundError:
        return False


def open_result(path, mode="rt"):
    path = result_path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=None if "b" in mode else "utf-8")
    return path.open(mode, encoding=None if "b" in mode else "utf-8")


def load_json(path):
    with open_result(path) as stream:
        return json.load(stream)


def write_text(path, text):
    """LF endings on every OS, so outputs are byte-stable across hosts."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_json(path, value, *, trailing_newline=False):
    write_text(path, json.dumps(value, indent=2) + ("\n" if trailing_newline else ""))


def open_csv(path):
    return Path(path).open("w", encoding="utf-8", newline="")


def add_paths(parser):
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Input evidence directory; .gz inputs are supported")
    parser.add_argument("--output-dir", type=Path,
                        help="Generated output directory; experiments/ is protected")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                        help="Experiment name in core/experiments or a path to its JSON")


def output_dir(value, *, default_temp=True):
    if value is None:
        if not default_temp:
            return None
        value = Path(tempfile.mkdtemp(prefix="siliconbadgers-profile-"))
    value = value.resolve()
    if value == PROTECTED.resolve() or PROTECTED.resolve() in value.parents:
        raise ValueError("experiments/ holds recorded evidence and is immutable; choose another output directory")
    value.mkdir(parents=True, exist_ok=True)
    return value


def parse_paths(description, *, default_temp=True, argv=None):
    parser = argparse.ArgumentParser(description=description)
    add_paths(parser)
    args = parser.parse_args(argv)
    args.results_dir = args.results_dir.resolve()
    args.output_dir = output_dir(args.output_dir, default_temp=default_temp)
    args.experiment = load_experiment(args.experiment)
    return args
