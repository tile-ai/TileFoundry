"""Meshes and operand layouts of the Metal SIMD-group instructions.

One instruction call covers G SIMD groups of one threadgroup. The matrix lane
map is unspecified by MSL and measured on Apple GPU family 8:
row = ((lane >> 1) & 3) + ((lane >> 4) & 1) * 4, col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4,
and each lane holds the elements (row, col) and (row, col + 1).
"""

from tilefoundry.ir import expr, pattern, types

G = pattern.Wildcard("groups", predicates=(pattern.Range(lo=1, hi=32),))
W = pattern.Wildcard("width", predicates=(pattern.Or(4, 32),))

MATRIX_FLOATS = (expr.DType.f16, expr.DType.bf16, expr.DType.f32)
LANE_INTS = (expr.DType.i8, expr.DType.u8, expr.DType.i16, expr.DType.u16, expr.DType.i32, expr.DType.u32)
LANE_NUMBERS = (expr.DType.f16, expr.DType.f32, *LANE_INTS)

# 32 lanes as four binary/quaternary digits: lane = d0 + 2*d1 + 8*d2 + 16*d3
SIMDGROUPS = pattern.Mesh(("thread",), pattern.Layout(((G, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),)))
# rows = (d3, d1), columns = (d2, d0, slot): the measured lane map above
FRAGMENT = pattern.ShardLayout(
    pattern.Layout((G, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))),
    (types.Split(0), types.Split(4), types.Split(2), types.Split(3), types.Split(1)),
    SIMDGROUPS,
)
# SIMD groups (width 32) or quad groups (width 4), one scalar per lane
LANE_GROUPS = pattern.Mesh(("thread",), pattern.Layout(((G, W),), ((W, 1),), predicates=(G * W <= 1024,)))
LANES = pattern.ShardLayout(pattern.Layout((G, W), (0, 0)), (types.Split(0), types.Split(1)), LANE_GROUPS)
SHARED = pattern.ShardLayout(pattern.Layout((G, 1), (0, 0)), (types.Split(0), types.Broadcast()), LANE_GROUPS)


def fragment() -> pattern.Tensor:
    """G 8x8 matrices held in registers, one per SIMD group."""
    return pattern.Tensor(shape=(G, 8, 8), dtype=pattern.Wildcard("dtype"), storage=expr.StorageKind.RMEM, layout=FRAGMENT)


def memory_matrices() -> pattern.Tensor:
    """G row-major 8x8 windows of device or threadgroup memory; the row stride is elements_per_row."""
    row = pattern.Wildcard("row_stride", predicates=(pattern.Range(lo=8),))
    return pattern.Tensor(
        shape=(G, 8, 8),
        dtype=pattern.Wildcard("dtype"),
        storage=pattern.Or(expr.StorageKind.GMEM, expr.StorageKind.SMEM),
        layout=pattern.Layout((G, 8, 8), (pattern.Wildcard("group_stride"), row, 1)),
    )


def lanes(dtype=pattern.Wildcard("dtype")) -> pattern.Tensor:
    """One scalar per lane."""
    return pattern.Tensor(shape=(G, W), dtype=dtype, storage=expr.StorageKind.RMEM, layout=LANES)


def shared(dtype=pattern.Wildcard("dtype")) -> pattern.Tensor:
    """One scalar per group, the same in every lane."""
    return pattern.Tensor(shape=(G, 1), dtype=dtype, storage=expr.StorageKind.RMEM, layout=SHARED)


def one_of(name: str, values) -> pattern.Wildcard:
    """A wildcard admitting exactly *values*."""
    return pattern.Wildcard(name, predicates=(pattern.Or(*values),))


__all__ = [
    "FRAGMENT", "G", "LANES", "LANE_GROUPS", "LANE_INTS", "LANE_NUMBERS", "MATRIX_FLOATS", "SHARED", "SIMDGROUPS", "W",
    "fragment", "lanes", "memory_matrices", "one_of", "shared",
]
