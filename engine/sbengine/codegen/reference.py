"""Value-level reference for checking generated matmul plans (issue 28, work item C).

Reproduces the declared arithmetic of the SoC projection experiment
(soc `sim/workload_probe.cpp`, `quantize` and `replace`):

* One symmetric INT8 scale per weight row and per activation vector:
  `scale = peak / 127` in float32, or 1.0 when every value is zero.
* `q = clamp(lround(value / scale), -127, 127)`: float32 division, halves
  rounded away from zero, and -128 is never produced.
* INT8 x INT8 products summed exactly. The probe sums each 8-element K tile in
  INT32 and combines tiles on the host in int64, so no sum wraps.
* `output = float(sum) * a_scale * b_scale`, each step rounded to float32,
  left to right.

This is the SoC experiment's test conversion, not the GGUF's own arithmetic,
an accepted model quantization scheme or the proposed Matrix unit's
requantization, which is not specified yet. Python floats are float64, so
every float32 step is rounded explicitly with `f32`.

Layout follows GGML and the probe: A is M weight rows of K values and B is N
activation vectors of K values, both K-innermost. Results are indexed [m][n]
for the logical C = A * B (A is M x K, B is K x N).
"""
import argparse
import hashlib
import json
import math
import random
import struct
import sys

SCHEMA = "siliconbadgers.compiler-reference.v0"
ARITHMETIC = "soc-workload-probe-int8-row-scale"
GENERATOR = "python-random-random-v1"
INT8_LIMIT = 127
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def f32(value):
    """Round a Python float to the nearest float32 (ties to even), overflowing to
    infinity as C float arithmetic does."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:  # raised only when the rounded value exceeds float32
        return math.copysign(math.inf, value)


def _round_half_away(value):
    """C `lround`: nearest integer, halves away from zero. Exact for any float."""
    whole = math.trunc(value)
    if abs(value - whole) >= 0.5:
        whole += 1 if value > 0 else -1
    return whole


def quantize_vector(values):
    """Quantize one weight row or activation vector; returns (q values, scale)."""
    values = [float(v) for v in values]
    if not values:
        raise ValueError("cannot quantize an empty vector")
    if any(not math.isfinite(v) or f32(v) != v for v in values):
        raise ValueError("inputs must be finite float32 values")
    peak = max(abs(v) for v in values)
    # float32 / float32 computed in float64 then rounded is the correctly
    # rounded float32 quotient, matching `peak / 127.0f` and `value / scale`.
    scale = f32(peak / 127.0) if peak > 0 else 1.0
    q = tuple(max(-INT8_LIMIT, min(INT8_LIMIT, _round_half_away(f32(v / scale)))) for v in values)
    return q, scale


def quantize_operands(a_rows, b_cols):
    """Quantize A rows and B columns; every vector must have the same K."""
    if not a_rows or not b_cols:
        raise ValueError("A and B need at least one vector each")
    k = len(a_rows[0])
    if any(len(v) != k for v in (*a_rows, *b_cols)):
        raise ValueError("every A row and B column must have the same K")
    a = [quantize_vector(row) for row in a_rows]
    b = [quantize_vector(col) for col in b_cols]
    return [q for q, _ in a], [s for _, s in a], [q for q, _ in b], [s for _, s in b]


def _check_int8(vectors, name):
    for vector in vectors:
        if any(type(v) is not int or not -128 <= v <= 127 for v in vector):
            raise ValueError(f"{name} values must be signed INT8 integers")


def check_operands(a_q, b_q):
    """Raise unless A and B are nonempty INT8 vectors that agree on K."""
    if not a_q or not b_q or not a_q[0]:
        raise ValueError("A and B need at least one nonempty vector each")
    _check_int8(a_q, "A")
    _check_int8(b_q, "B")
    if any(len(v) != len(a_q[0]) for v in (*a_q, *b_q)):
        raise ValueError("every A row and B column must have the same K")


def _sums(a_q, b_q, k0, stop):
    return [[sum(x * y for x, y in zip(row[k0:stop], col[k0:stop])) for col in b_q] for row in a_q]


def int_matmul(a_q, b_q, k0=0, k=None):
    """Exact integer sums C[m][n] over K indices [k0, k0 + k); no wrapping."""
    check_operands(a_q, b_q)
    full = len(a_q[0])
    k = full - k0 if k is None else k
    if k0 < 0 or k < 1 or k0 + k > full:
        raise ValueError("K range lies outside the operands")
    return _sums(a_q, b_q, k0, k0 + k)


def dequantize(sums, a_scales, b_scales):
    """float(sum) * a_scale * b_scale, rounded to float32 after each step."""
    return [[f32(f32(f32(float(s)) * a_scales[m]) * b_scales[n]) for n, s in enumerate(row)]
            for m, row in enumerate(sums)]


def soc_projection(a_rows, b_cols, k_tile=8):
    """The probe's full calculation, with its per-K-tile INT32 partial sums."""
    a_q, a_scales, b_q, b_scales = quantize_operands(a_rows, b_cols)
    k = len(a_q[0])
    sums = [[0] * len(b_q) for _ in a_q]
    for k0 in range(0, k, k_tile):
        partial = int_matmul(a_q, b_q, k0, min(k_tile, k - k0))
        for m, row in enumerate(partial):
            for n, value in enumerate(row):
                if not INT32_MIN <= value <= INT32_MAX:
                    raise ValueError("a K-tile partial sum exceeds INT32")
                sums[m][n] += value
    return {"a_q": a_q, "a_scales": a_scales, "b_q": b_q, "b_scales": b_scales,
            "sums": sums, "output": dequantize(sums, a_scales, b_scales)}


def digest(value):
    """SHA-256 of canonical JSON, for inputs too large to commit."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_vectors(seed, count, k, amplitude, outliers=0):
    """Deterministic float32 vectors from `random.random()`, whose sequence Python
    guarantees across versions. `outliers` channels per vector are scaled by 8,
    like the large activation channels in real LLM layers."""
    rng = random.Random(seed)
    vectors = []
    for _ in range(count):
        vector = [f32((2 * rng.random() - 1) * amplitude) for _ in range(k)]
        for _ in range(outliers):
            index = int(rng.random() * k)
            vector[index] = f32(vector[index] * 8)
        vectors.append(vector)
    return vectors


# Hand-chosen inputs. Values are exact float32 numbers.
_INLINE = {
    "hand_2x2x3": {
        "purpose": "Small enough to check by hand: scale 1.0, a half rounded away from zero "
                   "(-63.5 -> -64) and an all-zero vector (scale 1.0 path).",
        "a_rows": [[127.0, -63.5, 1.0], [0.0, 0.0, 0.0]],
        "b_cols": [[1.0, 2.0, 3.0], [-127.0, 127.0, 0.0]],
    },
    "extremes_4x3x16": {
        "purpose": "Peaks of either sign, a constant row, one dominant outlier that rounds "
                   "the rest to zero, tiny values, and a vector that is all zeros.",
        "a_rows": [
            [-2.0] + [1.0] * 15,
            [0.5] * 16,
            [1000.0] + [0.75 * (-1) ** i for i in range(15)],
            [2.0 ** -20 * (i - 8) for i in range(16)],
        ],
        "b_cols": [
            [float(i - 8) for i in range(16)],
            [0.0] * 16,
            [-3.0] * 8 + [3.0] * 8,
        ],
    },
}

# Seeded inputs, regenerated from the seed instead of committed.
_GENERATED = {
    "fixture_5x3x7": {
        "purpose": "Same logical shape as compiler-matmul.json (5 x 7 times 7 x 3).",
        "m": 5, "n": 3, "k": 7, "a": (101, 0.1, 0), "b": (102, 4.0, 0),
    },
    "edge_tiles_11x19x21": {
        "purpose": "No dimension is a multiple of the 8/16/8 placeholder tile, so every "
                   "tiling has edge tiles in M, N and K.",
        "m": 11, "n": 19, "k": 21, "a": (201, 0.1, 0), "b": (202, 4.0, 1),
    },
    "soc_prefill_8x7x2048": {
        "purpose": "SoC experiment prefill shape: first eight output channels, seven "
                   "prompt tokens, the full 2,048-element input dimension. Synthetic values.",
        "m": 8, "n": 7, "k": 2048, "a": (301, 0.05, 0), "b": (302, 2.0, 4),
    },
    "soc_decode_8x1x2048": {
        "purpose": "SoC experiment decode shape: one generated token. Synthetic values.",
        "m": 8, "n": 1, "k": 2048, "a": (401, 0.05, 0), "b": (402, 2.0, 4),
    },
}

# Inline cases embed every quantized value; larger ones embed digests only.
_EMBED_LIMIT = 512


def case_inputs(name, spec=None):
    """Return (a_rows, b_cols) for a named case, regenerating seeded inputs."""
    if name in _INLINE:
        return _INLINE[name]["a_rows"], _INLINE[name]["b_cols"]
    spec = _GENERATED[name] if spec is None else spec
    a_seed, a_amp, a_out = spec["a"]
    b_seed, b_amp, b_out = spec["b"]
    return (generate_vectors(a_seed, spec["m"], spec["k"], a_amp, a_out),
            generate_vectors(b_seed, spec["n"], spec["k"], b_amp, b_out))


def build_case(name):
    a_rows, b_cols = case_inputs(name)
    m, n, k = len(a_rows), len(b_cols), len(a_rows[0])
    result = soc_projection(a_rows, b_cols)
    case = {"name": name, "shape": {"m": m, "n": n, "k": k}}
    if name in _INLINE:
        case["purpose"] = _INLINE[name]["purpose"]
        case["inputs"] = {"kind": "inline", "a_rows": a_rows, "b_cols": b_cols}
    else:
        spec = _GENERATED[name]
        case["purpose"] = spec["purpose"]
        case["inputs"] = {
            "kind": "generated", "generator": GENERATOR,
            "a": dict(zip(("seed", "amplitude", "outliers"), spec["a"])),
            "b": dict(zip(("seed", "amplitude", "outliers"), spec["b"])),
            "a_rows_sha256": digest(a_rows), "b_cols_sha256": digest(b_cols),
        }
    expected = {"a_scales": result["a_scales"], "b_scales": result["b_scales"],
                "a_q_sha256": digest(result["a_q"]), "b_q_sha256": digest(result["b_q"]),
                "sums": result["sums"], "output": result["output"]}
    if (m + n) * k <= _EMBED_LIMIT:
        expected["a_q"], expected["b_q"] = result["a_q"], result["b_q"]
    case["expected"] = expected
    return case


def build_fixture():
    return {
        "schema": SCHEMA,
        "arithmetic": ARITHMETIC,
        "status": "reference_only",
        "source": "soc sim/workload_probe.cpp quantize/replace; experiment "
                  "experiments/2026-10-13-hls-platform",
        "limits": [
            "SoC test conversion only: not the GGUF arithmetic, an accepted model "
            "quantization or the proposed Matrix unit's requantization",
            "inputs are synthetic float32 values, not model weights or activations",
            "expected values come from this Python reference; they are not device results",
        ],
        "layout": "A is M rows of K, B is N columns of K (K innermost); sums and output are [m][n]",
        "cases": [build_case(name) for name in (*_INLINE, *_GENERATED)],
    }


def fixture_json():
    return json.dumps(build_fixture(), indent=1, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print the compiler reference fixture as JSON (stdout).")
    parser.parse_args(argv)
    sys.stdout.write(fixture_json())


if __name__ == "__main__":
    main()
