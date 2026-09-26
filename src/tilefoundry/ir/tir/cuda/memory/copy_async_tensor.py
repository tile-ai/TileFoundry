"""SM90 tensor-map asynchronous copy declaration and operand patterns."""

from __future__ import annotations

from enum import Enum

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    AndPattern,
    BitsPattern,
    CapturePattern,
    ComposedLayoutPattern,
    DistinctConstraint,
    LayoutPattern,
    MeshPattern,
    MultipleOfPattern,
    OrPattern,
    RangePattern,
    SameModesConstraint,
    SwitchPattern,
    SwizzlePattern,
    utils,
)
from tilefoundry.ir.pattern import (
    predicates as P,
)
from tilefoundry.ir.tir.verify import verify_between, verify_operands
from tilefoundry.ir.types import ComposedLayout, Layout, Mesh, Swizzle, UnitType
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

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


class BoxFamily(SwitchPattern):
    """Every unswizzled or swizzled shared-memory box."""

    def refusal(self, subject, captures=None) -> str | None:
        layout = subject
        transform = layout.inner if isinstance(layout, ComposedLayout) else None
        if transform is not None and layout.offset != 0:
            return f"it is reached through {transform!r} at offset {layout.offset}, not 0"
        for mode, pattern in self.branches:
            if mode.swizzle == transform:
                inner = pattern if transform is None else pattern.outer
                return inner.refusal(
                    layout if transform is None else layout.outer,
                    captures,
                )
        written = ", ".join(
            repr(mode.swizzle) for mode, _ in self.branches if mode.swizzle is not None
        )
        return f"it is reached through {transform!r}, and a tensormap swizzles by {written} or not at all"


def _dim(name: str, dtype: str, *, span: int | None = None) -> CapturePattern:
    parts = [
        RangePattern(lo=1, hi=BOX_EXTENT),
        BitsPattern(dtype, MultipleOfPattern(TMA_UNIT_BITS)),
    ]
    if span is not None:
        parts.append(BitsPattern(dtype, RangePattern(hi=span * 8)))
    return CapturePattern(name, AndPattern(tuple(parts)))


def TmaBoxPattern(dtype: str) -> BoxFamily:
    rest = tuple(
        CapturePattern(f"dim{index}", RangePattern(lo=1, hi=BOX_EXTENT))
        for index in range(1, TMA_RANK)
    )
    boxes = {
        TmaSwizzle.NONE: LayoutPattern(
            predicates=(
                P.PlainArrangement(),
                P.BoxDims((_dim("dim0", dtype), *rest), dtype, BOX_EXTENT),
            )
        )
    }
    for index, mode in enumerate(
        (mode for mode in TmaSwizzle if mode.swizzle is not None),
        TMA_RANK,
    ):
        swizzle = mode.swizzle
        boxes[mode] = ComposedLayoutPattern(
            SwizzlePattern(swizzle.bits, swizzle.base, swizzle.shift),
            0,
            LayoutPattern(
                predicates=(
                    P.PlainArrangement(),
                    P.BoxDims(
                        (_dim(f"dim{index}", dtype, span=mode.value), *rest),
                        dtype,
                        BOX_EXTENT,
                        span=mode.value,
                    ),
                )
            ),
        )
    return BoxFamily(SWIZZLE_PLACE, boxes)


def TmaGlobalPattern(dtype: str) -> LayoutPattern:
    steps = tuple(
        CapturePattern(
            f"step{index}",
            BitsPattern(dtype, MultipleOfPattern(TMA_UNIT_BITS)),
        )
        for index in range(1, TMA_RANK)
    )
    return LayoutPattern(predicates=(P.PlainArrangement(), P.TensorMap(steps, dtype)))


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
        offset=CapturePattern("p0", MultipleOfPattern(32)),
        outer=layout,
    )
    return MeshPattern(("thread",), OrPattern(sliced, layout))


@register_op(dialect="T", category="async", name="copy_async_tensor")
class CopyAsyncTensor(Op):
    """Move one tensor-map box between global and shared memory."""

    capability = "tma"

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
    scope = ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=_warp_scope(),
        optional=True,
        default=None,
    )


@register_typeinfer(CopyAsyncTensor)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(CopyAsyncTensor)
def verify_copy_async_tensor(call: "Call", ctx: "VerifyContext") -> None:
    src, dst = tuple(ctx.type_of(arg) for arg in call.args)
    if tuple(src.shape) != tuple(dst.shape) or src.dtype != dst.dtype:
        ctx.error(
            call,
            f"copy_async_tensor moves one tile: src is {tuple(src.shape)} "
            f"{src.dtype.name} and dst is {tuple(dst.shape)} {dst.dtype.name}",
        )
    moves = " and ".join(map(str, TMA_STORAGES))
    verify_between(
        call,
        ctx,
        f"copy_async_tensor moves a tile between {moves}, one end each: ",
    )
    verify_operands(call, ctx, "copy_async_tensor")


__all__ = [
    "BOX_EXTENT",
    "BoxFamily",
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
