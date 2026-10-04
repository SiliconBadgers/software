"""Storage formats: bytes per element, and the MAC-rate class a weight format falls into.

Block sizes are ggml's. They are checked against the captured graphs: token_embd.weight is q6_K,
2048 x 248320 elements = 508,559,360 -> 508,559,360 / 256 * 210 = 417,177,600 bytes, which is the
`logical_bytes` recorded in the capture (tests/test_formats.py).
"""
import math

# dtype -> (elements per block, bytes per block)
BLOCK = {
    "f32": (1, 4), "f16": (1, 2), "bf16": (1, 2), "i32": (1, 4), "i64": (1, 8), "i8": (1, 1),
    "q4_0": (32, 18), "q8_0": (32, 34),
    "q4_K": (256, 144), "q5_K": (256, 176), "q6_K": (256, 210),
}


def storage_bytes(dtype, elements):
    """Bytes to store `elements` values of `dtype` (whole blocks)."""
    if dtype not in BLOCK:
        raise KeyError(f"unknown dtype {dtype!r}; add it to formats.BLOCK")
    per_block, block_bytes = BLOCK[dtype]
    return math.ceil(elements / per_block) * block_bytes


def bits_per_element(dtype):
    per_block, block_bytes = BLOCK[dtype]
    return 8.0 * block_bytes / per_block


def rate_class(dtype):
    """Weight-format class used to look up a matrix MAC rate (MACs per PE per cycle).

    Block-quantized weights are assumed to be dequantized inside the datapath, so the class follows the
    bit width: <=4.5 bit -> w4, <=5.5 -> w5, <=6.6 -> w6, <=8.5 -> w8, 16-bit floats -> w16, f32 -> w32.
    The class -> rate table lives in the hardware config (matrix.rates); it is an assumption, not a measurement.
    """
    bits = bits_per_element(dtype)
    if dtype == "f32":
        return "w32"
    if bits <= 4.6:
        return "w4"
    if bits <= 5.6:
        return "w5"
    if bits <= 6.9:
        return "w6"
    if bits <= 8.6:
        return "w8"
    return "w16"
