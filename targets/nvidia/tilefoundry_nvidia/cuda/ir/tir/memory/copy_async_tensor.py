"""SM90 tensor-map asynchronous copy declaration and operand patterns."""

from __future__ import annotations

from enum import Enum

from tilefoundry.interfaces import access, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.tir import memory

TMA_RANK = 5
BOX_EXTENT = 256
TMA_UNIT_BITS = 16 * 8
SWIZZLE_PLACE = "swizzle"
TMA_STORAGES = (expr.StorageKind.GMEM, expr.StorageKind.SMEM)


class TmaSwizzle(Enum):
    """The shared-memory swizzle selected when the tensor map is encoded."""

    NONE = 0
    SW32 = 32
    SW64 = 64
    SW128 = 128

    @property
    def swizzle(self) -> types.Swizzle | None:
        return None if self is TmaSwizzle.NONE else types.Swizzle(self.value.bit_length() - 5, 4, 3)


def dim_wildcard(name: str) -> pattern.Wildcard:
    return pattern.Wildcard(name)


def dim_formulas(name: str, dtype: str, *, span: int | None = None) -> tuple:
    dim = pattern.Wildcard(name)
    parts = [*dim_range(name), pattern.Bits(dtype) * dim % TMA_UNIT_BITS == 0]
    if span is not None:
        parts.append(pattern.Bits(dtype) * dim <= span * 8)
    return tuple(parts)


def dim_range(name: str) -> tuple:
    dim = pattern.Wildcard(name)
    return dim >= 1, dim <= BOX_EXTENT


def tma_box_pattern(dtype: str, *, allow_partial=False) -> pattern.Switch:
    rest = tuple(dim_wildcard(f"dim{index}") for index in range(1, TMA_RANK))
    rest_formulas = tuple(
        formula for index in range(1, TMA_RANK) for formula in dim_range(f"dim{index}")
    )
    plain_dims = (dim_wildcard("dim0"), *rest)
    boxes = {
        TmaSwizzle.NONE: pattern.Layout(
            predicates=(
                pattern.PlainArrangement(allow_partial=allow_partial),
                pattern.BoxDims(plain_dims, dtype, BOX_EXTENT, allow_partial=allow_partial),
                *dim_formulas("dim0", dtype),
                *rest_formulas,
            )
        )
    }
    for index, mode in enumerate(
        (mode for mode in TmaSwizzle if mode.swizzle is not None),
        TMA_RANK,
    ):
        swizzle = mode.swizzle
        leading = f"dim{index}"
        dims = (dim_wildcard(leading), *rest)
        boxes[mode] = pattern.ComposedLayout(
            pattern.Swizzle(swizzle.bits, swizzle.base, swizzle.shift),
            0,
            pattern.Layout(
                predicates=(
                    pattern.PlainArrangement(allow_partial=allow_partial),
                    pattern.BoxDims(
                        dims,
                        dtype,
                        BOX_EXTENT,
                        span=mode.value,
                        allow_partial=allow_partial,
                    ),
                    *dim_formulas(leading, dtype, span=mode.value),
                    *rest_formulas,
                )
            ),
        )
    return pattern.Switch(SWIZZLE_PLACE, boxes)


def tma_global_pattern(dtype: str) -> pattern.Layout:
    steps = tuple(pattern.Wildcard(f"step{index}") for index in range(1, TMA_RANK))
    return pattern.Layout(
        predicates=(
            pattern.PlainArrangement(),
            pattern.TensorMap(steps, dtype),
            *(pattern.Bits(dtype) * step % TMA_UNIT_BITS == 0 for step in steps),
        )
    )


def tma_operand_pattern(storage: str, dtype: str) -> pattern.Switch:
    return pattern.Switch(
        storage,
        {
            expr.StorageKind.GMEM: tma_global_pattern(dtype),
            expr.StorageKind.SMEM: tma_box_pattern(dtype),
        },
    )


def warp_scope() -> pattern.Mesh:
    layout = pattern.Layout(
        ((pattern.Star(pattern.Or(1)), 32),),
        ((pattern.Star(pattern.Wildcard()), 1),),
    )
    sliced = pattern.ComposedLayout(
        inner=None,
        offset=pattern.Wildcard("p0"),
        outer=layout,
        predicates=(pattern.Wildcard("p0") % 32 == 0, pattern.Forward(), pattern.Injective()),
    )
    return pattern.Mesh(("thread",), sliced)


@expr.register_op(dialect="T", category="async", name="copy_async_tensor")
class CopyAsyncTensor(expr.Op):
    """Move one tensor-map box between global and shared memory."""

    capability = expr.OpCapability("cp.async.bulk.tensor")
    resource = "tma_engine"

    src = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        pattern=pattern.operand_tile(
            0,
            TMA_STORAGES,
            tma_operand_pattern(pattern.storage_place(0), pattern.dtype_place(0)),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.WRITE,
        pattern=pattern.operand_tile(
            1,
            TMA_STORAGES,
            tma_operand_pattern(pattern.storage_place(1), pattern.dtype_place(1)),
        ),
    )
    between = (
        pattern.SameConstraint("shape", "src", "dst"),
        pattern.SameConstraint("dtype", "src", "dst"),
        pattern.DistinctConstraint("storage", "src", "dst"),
        pattern.SameModesConstraint("src", "dst"),
    )
    execution_mesh = expr.ParamDef(
        kind="attribute",
        annotation=types.Mesh,
        pattern=warp_scope(),
        optional=True,
        default=None,
    )


memory.register_copy_semantics(CopyAsyncTensor, verify=False)


@expr.register_op(dialect="T", category="async", name="copy_async_tensor_tx")
class CopyAsyncTensorTx(expr.Op):
    """Load one tensor-map box into shared memory, completing its bytes on an mbarrier word.

    Lowering emits it for a selected ``CopyAsyncTensor`` from global to shared
    memory; a schedule never names it. ``barrier`` holds one word per warp; the
    first issuing thread completes the load on its own warp's word. The box is
    in ``dst`` for every issuing thread when the operation ends.
    """

    src = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        pattern=pattern.operand_tile(
            0,
            expr.StorageKind.GMEM,
            tma_operand_pattern(pattern.storage_place(0), pattern.dtype_place(0)),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.WRITE,
        pattern=pattern.operand_tile(
            1,
            expr.StorageKind.SMEM,
            tma_operand_pattern(pattern.storage_place(1), pattern.dtype_place(1)),
        ),
    )
    barrier = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ | expr.MemoryEffect.WRITE,
        pattern=pattern.tensor_in(expr.StorageKind.SMEM),
    )
    between = (
        pattern.SameConstraint("shape", "src", "dst"),
        pattern.SameConstraint("dtype", "src", "dst"),
        pattern.SameModesConstraint("src", "dst"),
    )
    execution_mesh = expr.ParamDef(
        kind="attribute",
        annotation=types.Mesh,
        pattern=warp_scope(),
        optional=True,
        default=None,
    )


@typeinfer.register(CopyAsyncTensorTx)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


@access.register_relation(CopyAsyncTensorTx)
def tensor_tx_access(call, ctx):
    source = ctx.type_of(call.args[0])
    data = access.identity(len(source.shape))
    barrier = access.state(ctx.type_of(call.args[2]), len(source.shape))
    return access.iterating(source.shape, (data, data, barrier, data))


__all__ = [
    "BOX_EXTENT",
    "CopyAsyncTensor",
    "CopyAsyncTensorTx",
    "SWIZZLE_PLACE",
    "TMA_RANK",
    "TMA_STORAGES",
    "TMA_UNIT_BITS",
    "tma_box_pattern",
    "tma_global_pattern",
    "tma_operand_pattern",
    "TmaSwizzle",
    "warp_scope",
]
