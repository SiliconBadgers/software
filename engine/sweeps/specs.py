"""Shared input contract for pinned studies; no evaluation or output generation."""
import json
import math
from pathlib import Path, PureWindowsPath
import re
import sys

from sbengine import config

# Numeric public summary fields, not labels, hashes or validity flags. Numeric
# declared axes can also be objectives. Keep this contract covered by summary tests.
NUMERIC_METRICS = frozenset((
    "cost", "dsp", "prefill_s", "decode_s", "prefill_tps", "decode_tps_per_seq", "decode_tps_total",
    "prefill_lower_s", "prefill_serial_s", "decode_lower_s", "decode_serial_s",
    "decode_l1_service_s", "decode_hbm_service_s", "hbm_required_gib",
))


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _finite_number(value):
    # Comparing integers to the finite range does not coerce/round them to float
    # (math.isfinite would overflow for large ints). Booleans are not metrics.
    return ((type(value) is int and -sys.float_info.max <= value <= sys.float_info.max)
            or (type(value) is float and math.isfinite(value)))


def _integer(value):
    number = int(value)
    if not _finite_number(number):
        raise ValueError("JSON integer exceeds the finite binary64 range")
    return number


def _float(value):
    number = float(value)
    if not _finite_number(number):
        _constant(value)
    # JSON has already recognized the number grammar. Only a nonzero
    # significand rounded to zero is underflow; literal zeros remain valid.
    if number == 0 and any(digit in "123456789" for digit in value.lower().partition("e")[0]):
        raise ValueError("nonzero JSON number underflows to zero")
    return number


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                       parse_constant=_constant, parse_float=_float, parse_int=_integer)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def repository_path(root, name):
    """Portable repository-relative reference, including symlink containment."""
    if (not isinstance(name, str) or not name or "\\" in name or PureWindowsPath(name).drive
            or Path(name).is_absolute()):
        raise ValueError(f"invalid repository reference: {name!r}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"escaping repository reference: {name}")
    return path


def reference(root, name):
    path = repository_path(root, name)
    if not path.is_file():
        raise ValueError(f"missing repository reference: {name}")
    return path


def _name(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) or value.endswith("."):
        raise ValueError(f"{field} must be a nonempty portable name")
    return value


def load_spec(root, path):
    """Validate declarations before captures, traces, enumeration or output use."""
    root = Path(root).resolve()
    spec = _read(Path(path))
    required = {"id", "baseline", "workload", "axes", "objectives", "decision", "prerequisites", "reference"}
    if missing := required - spec.keys():
        raise ValueError("missing study fields: " + ", ".join(sorted(missing)))
    _name(spec["id"], "study id")
    _name(spec["workload"], "workload")
    baseline = _read(reference(root, spec["baseline"]))
    cfg, profile = config.load_profile(reference(root, baseline["profile"]))
    if not profile.get("resolved") or config.config_hash(cfg) != baseline["config_hash"]:
        raise ValueError("study baseline must be a complete matching resolved profile")
    repository_path(root, baseline["run"])
    reference(root, spec["reference"])
    decisions = _read(reference(root, baseline["decisions"]))
    known = {item["id"] for item in decisions["branches"]}
    prerequisites = spec["prerequisites"]
    if (not isinstance(spec["decision"], str) or spec["decision"] not in known
            or not isinstance(prerequisites, list)
            or any(not isinstance(item, str) or item not in known for item in prerequisites)):
        raise ValueError("unknown or malformed decision/prerequisites")
    fixed = baseline["fixed_assumptions"]
    if not isinstance(fixed, list) or any(not isinstance(item, str) or not item for item in fixed):
        raise ValueError("fixed_assumptions must be a list of configuration paths")
    for key in fixed:
        config.get_path(cfg, key)
    axes = spec["axes"]
    if not isinstance(axes, dict):
        raise ValueError("axes must be an object of configuration paths and nonempty value lists")
    numeric_axes = set()
    for key, values in axes.items():
        if not key or not isinstance(values, list) or not values:
            raise ValueError(f"axis {key!r} must have a nonempty value list")
        config.get_path(cfg, key)
        if any(key == item or key.startswith(item + ".") or item.startswith(key + ".") for item in fixed):
            raise ValueError(f"axis {key} varies a fixed platform/external assumption")
        if any(other != key and other.startswith(key + ".") for other in axes):
            raise ValueError(f"overlapping configuration axes: {key}")
        if all(_finite_number(value) for value in values):
            numeric_axes.add(key)
    objectives = spec["objectives"]
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("objectives must be a nonempty list of [numeric metric, min/max] pairs")
    seen = set()
    for item in objectives:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("each objective must be a [numeric metric, min/max] pair")
        metric, direction = item
        if not isinstance(metric, str) or metric not in NUMERIC_METRICS | numeric_axes:
            raise ValueError(f"unsupported numeric objective metric: {metric!r}")
        if direction not in ("min", "max"):
            raise ValueError(f"objective {metric}: direction must be min or max")
        if metric in seen:
            raise ValueError(f"duplicate or conflicting objective: {metric}")
        seen.add(metric)
    return spec, baseline, cfg


def validate_objective_values(records, objectives):
    """Do not emit a frontier for nonnumeric/nonfinite feasible objective values."""
    for row in records:
        if row.get("valid"):
            for metric, _ in objectives:
                value = row.get(metric)
                if not _finite_number(value):
                    raise ValueError(f"out-of-range, non-finite or nonnumeric result for objective: {metric}")
