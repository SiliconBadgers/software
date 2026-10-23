"""Captured dense work conventions, not a total FLOP or hardware-time model."""


def dense_macs(op, elements, source_shape):
    """MUL_MAT reduction dimension is source 0; SSM_CONV kernel taps are source 1.

    source_shape(slot) adapts the caller's operand representation without copying
    its graph. Preserve the caller's captured slot/index convention.
    """
    slot = {"MUL_MAT": 0, "SSM_CONV": 1}.get(op)
    return None if slot is None else elements * source_shape(slot)[0]
