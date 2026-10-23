"""Pinned-study input identity, separate from graph interpretation and characterization."""
import hashlib
import json


def decoded_capture_sha256(capture):
    """Hash every parsed JSON field; object order, whitespace and compression are irrelevant."""
    canonical = json.dumps(capture, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_capture(graph, key, baseline):
    """Validate the decoded input that constructed this graph, without rereading its path."""
    try:
        if key not in baseline["captures"]:
            raise KeyError(key)
        expected = baseline["decoded_capture_sha256"][key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"capture identity missing: {key}") from error
    actual = decoded_capture_sha256(graph.capture)
    if actual != expected:
        raise ValueError(f"capture identity differs: {key} (decoded JSON) loaded from {graph.path}; "
                         f"expected {expected}, got {actual}")


def validate_captures(workload, baseline):
    for phase in ("prefill", "decode"):
        validate_capture(workload.graph(phase), f"{workload.name}/{phase}", baseline)
