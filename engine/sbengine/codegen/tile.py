"""Split one symbolic matmul into hardware-sized tiles in MATRIX run order.

Tiles are element offsets and sizes in the logical C = A*B (A is M x K, B is K x N),
not addresses or buffers. Tile limits are caller-supplied; the SoC projection's
8/16/8 is a placeholder until the target interface (issue 28, item B) is agreed.
"""
from dataclasses import dataclass

from .plan import MatmulPlan, _integer


@dataclass(frozen=True)
class MatmulTile:
    tensor_id: int
    m0: int
    n0: int
    k0: int
    m: int
    n: int
    k: int
    clear_acc: bool
    write_output: bool

    def validate(self):
        if not all(_integer(v) for v in (self.tensor_id, self.m0, self.n0, self.k0)):
            raise ValueError("tile identity and offsets must be nonnegative integers")
        if not all(_integer(v, 1) for v in (self.m, self.n, self.k)):
            raise ValueError("tile M/N/K must be positive integers")
        if type(self.clear_acc) is not bool or type(self.write_output) is not bool:
            raise ValueError("accumulator flags must be booleans")


@dataclass(frozen=True)
class TiledMatmul:
    plan: MatmulPlan
    tile_m: int
    tile_n: int
    tile_k: int

    def validate(self):
        if not isinstance(self.plan, MatmulPlan):
            raise ValueError("tiling requires a MatmulPlan")
        self.plan.validate()
        if not all(_integer(v, 1) for v in (self.tile_m, self.tile_n, self.tile_k)):
            raise ValueError("maximum tile sizes must be positive integers")

    def tiles(self):
        """Yield tiles m, then n, then k innermost: one C tile's K chunks run back to back.

        The first K chunk clears the accumulator and the last writes the output.
        Edge tiles are smaller when a dimension is not a multiple of its limit.
        """
        p = self.plan
        for m0 in range(0, p.m, self.tile_m):
            for n0 in range(0, p.n, self.tile_n):
                for k0 in range(0, p.k, self.tile_k):
                    k = min(self.tile_k, p.k - k0)
                    yield MatmulTile(p.tensor_id, m0, n0, k0,
                                     min(self.tile_m, p.m - m0), min(self.tile_n, p.n - n0), k,
                                     k0 == 0, k0 + k == p.k)
