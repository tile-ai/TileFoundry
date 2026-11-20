"""How ANE instructions see their operands.

A feature map is [1, C, H, W] in unified memory (GMEM) or in the on-chip L2
buffer (SMEM); W is contiguous. Weights stay in unified memory. A mesh layout
holds one mode per topology.
"""

from tilefoundry.ir import expr, pattern

ENGINE = pattern.Mesh(("engine",), pattern.Layout(((1,),), ((1,),)))
FLOATS = (expr.DType.f16,)
FEATURE_STORAGE = pattern.Or(expr.StorageKind.GMEM, expr.StorageKind.SMEM)

MAX_CHANNELS = 65536
MAX_HEIGHT = 16384
MAX_WIDTH = 16384
MAX_KERNEL = 32


def extent(name: str, limit: int) -> pattern.Wildcard:
    """An extent of at most *limit*, captured as *name*."""
    return pattern.Wildcard(name, predicates=(pattern.Range(hi=limit),))


def feature(operand: str) -> pattern.Tensor:
    """[1, C, H, W] in GMEM or SMEM, within the architecture limits.

    Extents and storage are captured under *operand*; every operand shares one dtype.
    """
    return pattern.Tensor(
        shape=(
            1,
            extent(f"{operand}.shape[1]", MAX_CHANNELS),
            extent(f"{operand}.shape[2]", MAX_HEIGHT),
            extent(f"{operand}.shape[3]", MAX_WIDTH),
        ),
        dtype=one_of("dtype", FLOATS),
        storage=pattern.Wildcard(f"{operand}.storage", predicates=(FEATURE_STORAGE,)),
    )


def one_of(name: str, values) -> pattern.Wildcard:
    """A wildcard admitting exactly *values*."""
    return pattern.Wildcard(name, predicates=(pattern.Or(*values),))


__all__ = [
    "ENGINE", "FEATURE_STORAGE", "FLOATS", "MAX_CHANNELS", "MAX_HEIGHT", "MAX_KERNEL", "MAX_WIDTH", "extent",
    "feature", "one_of",
]
