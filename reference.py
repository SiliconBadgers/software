"""Small independent signed-MAC reference; standard library only."""
def step(acc, a, b, enable=True, clear=False):
    if not (-128 <= a <= 127 and -128 <= b <= 127):
        raise ValueError('MAC operands must fit signed INT8')
    if clear:
        return 0
    if not enable:
        return acc
    unsigned = (acc + a*b) & 0xffffffff
    return unsigned - (1 << 32) if unsigned >= (1 << 31) else unsigned
