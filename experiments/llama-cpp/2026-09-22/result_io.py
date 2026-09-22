"""Portable readers and output guards for the recorded profiling evidence."""
import argparse
import gzip
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
RECORDED = ROOT / "results"


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
    return gzip.open(path, mode) if path.suffix == ".gz" else path.open(mode)


def load_json(path):
    with open_result(path) as stream:
        return json.load(stream)


def add_paths(parser):
    parser.add_argument("--results-dir", type=Path, default=RECORDED,
                        help="Input evidence directory; .gz inputs are supported")
    parser.add_argument("--output-dir", type=Path,
                        help="Generated output directory; recorded results are protected")


def output_dir(value, *, default_temp=True):
    if value is None:
        if not default_temp:
            return None
        value = Path(tempfile.mkdtemp(prefix="siliconbadgers-profile-"))
    value = value.resolve()
    if value == RECORDED.resolve() or RECORDED.resolve() in value.parents:
        raise ValueError("The packaged results directory is immutable; choose a separate output directory")
    value.mkdir(parents=True, exist_ok=True)
    return value


def parse_paths(description, *, default_temp=True):
    parser = argparse.ArgumentParser(description=description)
    add_paths(parser)
    args = parser.parse_args()
    args.results_dir = args.results_dir.resolve()
    args.output_dir = output_dir(args.output_dir, default_temp=default_temp)
    return args
