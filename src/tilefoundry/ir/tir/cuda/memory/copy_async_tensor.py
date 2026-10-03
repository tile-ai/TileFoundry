"""SM90 tensor-map asynchronous copy declaration and operand patterns."""

from __future__ import annotations

from enum import Enum

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    DistinctConstraint,
    LayoutPattern,
    MeshPattern,
    SameModesConstraint,
    SwitchPattern,
    SwizzlePattern,
    WildcardPattern,
    utils,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.types import Layout, Mesh, Swizzle, UnitType
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)

TMA_RANK = 5
BOX_EXTENT = 256
TMA_UNIT_BITS = 16 * 8
SWIZZLE_PLACE = "swizzle"
TMA_STORAGES = (S.GMEM, S.SMEM)


class TmaSwizzle(Enum):
    """The shared-memory swizzle selected when the tensor map is encoded."""

    NONE = 0
    SW32 = 32
    SW64 = 64
    SW128 = 128

    @property
    def swizzle(self) -> Swizzle | None:
        return None if self is TmaSwizzle.NONE else Swizzle(self.value.bit_length() - 5, 4, 3)


def _dim(name: str) -> WildcardPattern:
    return WildcardPattern(name)


def _dim_formulas(name: str, dtype: str, *, span: int | None = None) -> tuple:
    dim = WildcardPattern(name)
    parts = [*_dim_range(name), P.Bits(dtype) * dim % TMA_UNIT_BITS == 0]
    if span is not None:
        parts.append(P.Bits(dtype) * dim <= span * 8)
    return tuple(parts)


def _dim_range(name: str) -> tuple:
    dim = WildcardPattern(name)
    return dim >= 1, dim <= BOX_EXTENT


def TmaBoxPattern(dtype: str) -> SwitchPattern:
    rest = tuple(_dim(f"dim{index}") for index in range(1, TMA_RANK))
    rest_formulas = tuple(
        formula for index in range(1, TMA_RANK) for formula in _dim_range(f"dim{index}")
    )
    plain_dims = (_dim("dim0"), *rest)
    boxes = {
        TmaSwizzle.NONE: LayoutPattern(
            predicates=(
                P.PlainArrangement(),
                P.BoxDims(plain_dims, dtype, BOX_EXTENT),
                *_dim_formulas("dim0", dtype),
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
        dims = (_dim(leading), *rest)
        boxes[mode] = ComposedLayoutPattern(
            SwizzlePattern(swizzle.bits, swizzle.base, swizzle.shift),
            0,
            LayoutPattern(
                predicates=(
                    P.PlainArrangement(),
                    P.BoxDims(
                        dims,
                        dtype,
                        BOX_EXTENT,
                        span=mode.value,
                    ),
                    *_dim_formulas(leading, dtype, span=mode.value),
                    *rest_formulas,
                )
            ),
        )
    return SwitchPattern(SWIZZLE_PLACE, boxes)


def TmaGlobalPattern(dtype: str) -> LayoutPattern:
    steps = tuple(WildcardPattern(f"step{index}") for index in range(1, TMA_RANK))
    return LayoutPattern(
        predicates=(
            P.PlainArrangement(),
            P.TensorMap(steps, dtype),
            *(P.Bits(dtype) * step % TMA_UNIT_BITS == 0 for step in steps),
        )
    )


def TmaOperandPattern(storage: str, dtype: str) -> SwitchPattern:
    return SwitchPattern(
        storage,
        {
            S.GMEM: TmaGlobalPattern(dtype),
            S.SMEM: TmaBoxPattern(dtype),
        },
    )


def _warp_scope() -> MeshPattern:
    layout = LayoutPattern(
        ((32,),),
        ((1,),),
        predicates=(P.Forward(per_mode=True), P.Injective(per_mode=True)),
    )
    sliced = ComposedLayoutPattern(
        inner=None,
        offset=WildcardPattern("p0"),
        outer=layout,
        predicates=(WildcardPattern("p0") % 32 == 0,),
    )
    return MeshPattern(("thread",), sliced)


@register_op(dialect="T", category="async", name="copy_async_tensor")
class CopyAsyncTensor(Op):
    """Move one tensor-map box between global and shared memory."""

    capability = OpCapability("cp.async.bulk.tensor")
    resource = "tma_engine"

    src = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        pattern=utils.operand_tile(
            0,
            TMA_STORAGES,
            TmaOperandPattern(utils.storage_place(0), utils.dtype_place(0)),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=MemoryEffect.WRITE,
        pattern=utils.operand_tile(
            1,
            TMA_STORAGES,
            TmaOperandPattern(utils.storage_place(1), utils.dtype_place(1)),
        ),
    )
    between = (
        DistinctConstraint("storage", "src", "dst"),
        SameModesConstraint("src", "dst"),
    )
    smem_layout = ParamDef(
        kind="attribute",
        annotation=Layout,
        optional=True,
        default=None,
    )
    execution_mesh = ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=_warp_scope(),
        optional=True,
        default=None,
    )


@register_typeinfer(CopyAsyncTensor)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


register_access_relation(CopyAsyncTensor)(identity_relations)


@register_schedule_eval(CopyAsyncTensor)
def _eval_scheduled_copy_async_tensor(ctx):
    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)


@register_verify_stmt(CopyAsyncTensor)
def verify_copy_async_tensor(call: "Call", ctx: "VerifyContext") -> None:
    src, dst = tuple(ctx.type_of(arg) for arg in call.args)
    if tuple(src.shape) != tuple(dst.shape) or src.dtype != dst.dtype:
        ctx.error(
            call,
            f"copy_async_tensor moves one tile: src is {tuple(src.shape)} "
            f"{src.dtype.name} and dst is {tuple(dst.shape)} {dst.dtype.name}",
        )


__all__ = [
    "BOX_EXTENT",
    "CopyAsyncTensor",
    "SWIZZLE_PLACE",
    "TMA_RANK",
    "TMA_STORAGES",
    "TMA_UNIT_BITS",
    "TmaBoxPattern",
    "TmaGlobalPattern",
    "TmaOperandPattern",
    "TmaSwizzle",
]
