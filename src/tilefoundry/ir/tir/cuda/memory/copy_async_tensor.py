"""SM90 tensor-map asynchronous copy declaration and operand patterns."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    UNNAMED_PLACE,
    AndPattern,
    BitsPattern,
    CapturePattern,
    ComposedLayoutPattern,
    DistinctConstraint,
    LayoutPattern,
    MeshPattern,
    MultipleOfPattern,
    OrPattern,
    Pattern,
    RangePattern,
    SameModesConstraint,
    SequencePattern,
    SwitchPattern,
    SwizzlePattern,
    affine_part,
    matched,
    relations_of,
    utils,
)
from tilefoundry.ir.pattern.match import written_place, written_tuple
from tilefoundry.ir.tir.verify import verify_between, verify_operands
from tilefoundry.ir.types import ComposedLayout, Layout, Mesh, ShardLayout, Swizzle, UnitType
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

TMA_RANK = 5
BOX_EXTENT = 256
TMA_UNIT_BITS = 16 * 8
SWIZZLE_PLACE = "swizzle"
BOX_READING = (
    "every box: each tile axis's modes, contiguous ones joined up to "
    f"{BOX_EXTENT} elements, one dim each, in increasing step"
)
TENSORMAP_READING = (
    "every tensormap: one dim per mode of the tile, the mode at step 1 first"
)
TMA_STORAGES = (S.GMEM, S.SMEM)


@dataclass(frozen=True)
class Run:
    """One contiguous run of modes from one logical tile axis."""

    extent: int
    step: int
    axis: int
    mode: int


def box_runs(
    layout: Layout,
    element_bits: int,
    span: int | None,
    limit: int | None = BOX_EXTENT,
) -> tuple[Run, ...]:
    """Read TMA box runs by tile axis, ordered by increasing step."""
    runs: list[Run] = []
    for axis, (extents, steps) in enumerate(zip(layout.shape, layout.strides)):
        modes = tuple(enumerate(zip(flatten(extents), flatten(steps))))
        for mode, (extent, step) in reversed(modes):
            if extent == 1:
                continue
            last = runs[-1] if runs and runs[-1].axis == axis else None
            joined = None if last is None else last.extent * extent
            if (
                last is not None
                and step == last.step * last.extent
                and (limit is None or joined <= limit)
                and not (
                    span is not None
                    and last.step == 1
                    and joined * element_bits > span * 8
                )
            ):
                runs[-1] = replace(last, extent=joined)
            else:
                runs.append(Run(extent, step, axis, mode))
    return tuple(sorted(runs, key=lambda run: run.step))


class TmaSwizzle(Enum):
    """The shared-memory swizzle selected when the tensor map is encoded."""

    NONE = 0
    SW32 = 32
    SW64 = 64
    SW128 = 128

    @property
    def swizzle(self) -> Swizzle | None:
        return None if self is TmaSwizzle.NONE else Swizzle(self.value.bit_length() - 5, 4, 3)


def _missed_place(places, values, captures, first: int = 0) -> tuple | None:
    held = captures
    for index, (place, value) in enumerate(zip(places, values), first):
        found = matched(place, value, held)
        if found is None:
            return index, place, value
        held = found.captures
    return None


@dataclass(frozen=True)
class BoxPattern(Pattern):
    """One TMA box landed in shared memory."""

    dims: tuple
    dtype: str
    span: int | None = None

    def reading(self, subject, captures) -> tuple[tuple | None, str | None]:
        layout = affine_part(subject, plain=True)
        if layout is None or any(
            type(value) is not int
            for group in (layout.shape, layout.strides)
            for value in flatten(group)
        ):
            return None, f"{subject!r} is no static strided arrangement"
        width = getattr(captures.get(self.dtype), "bit_width", None)
        if type(width) is not int:
            return None, f"the element it arranges is not bound as {self.dtype}"
        runs = box_runs(layout, width, self.span)
        if not runs:
            return None, "it holds one element, which is no box"
        if len(runs) > len(self.dims):
            return None, (
                f"it is {len(runs)} runs of modes, and a box has at most "
                f"{len(self.dims)} dims"
            )
        extents = tuple(run.extent for run in runs)
        extents += (1,) * (len(self.dims) - len(extents))
        if runs[0].step != 1:
            return extents, (
                f"its smallest step is {runs[0].step}, and a box lays its dim 0 at step 1"
            )
        if self.span is not None and (self.span * 8) % width:
            return extents, f"a {self.span}-byte row is no whole number of {self.dtype}"
        expected = runs[0].extent if self.span is None else self.span * 8 // width
        for index, run in enumerate(runs[1:], 1):
            if run.step != expected:
                return extents, (
                    f"its dim {index} steps {run.step} where a box lays it at "
                    f"{expected} ({self.laid()})"
                )
            expected *= run.extent
        return extents, None

    def laid(self) -> str:
        rows = "" if self.span is None else f", rows {self.span} B apart"
        return f"dim 0 fastest{rows}"

    def match(self, subject, captures=None):
        held = dict(captures or {})
        extents, unlaid = self.reading(subject, held)
        if extents is None or unlaid is not None:
            return None
        return matched(SequencePattern(*self.dims), extents, held)

    def refusal(self, subject, captures=None) -> str | None:
        held = dict(captures or {})
        extents, why = self.reading(subject, held)
        if extents is None:
            return why
        missed = _missed_place(self.dims, extents, held)
        if missed is not None:
            index, place, extent = missed
            return (
                f"its box dim {index} holds {extent} elements, and a box reads "
                f"{written_place(place.pattern, place.name)}"
            )
        return why

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        dims = written_tuple(tuple(written_place(place) for place in self.dims))
        return f"box {dims}, {self.laid()}"

    def relations(self) -> tuple[str, ...]:
        return (BOX_READING, *relations_of(self.dims))


@dataclass(frozen=True, init=False)
class BoxFamily(SwitchPattern):
    """Every unswizzled or swizzled shared-memory box."""

    def match(self, subject, captures=None):
        if isinstance(subject, ShardLayout) and affine_part(subject) is not None:
            subject = subject.layout
        return super().match(subject, captures)

    def refusal(self, subject, captures=None) -> str | None:
        layout = subject
        if isinstance(layout, ShardLayout) and affine_part(layout) is not None:
            layout = layout.layout
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


@dataclass(frozen=True)
class TensorMapPattern(Pattern):
    """The global tile described by one tensor map."""

    steps: tuple
    dtype: str

    def reading(self, subject) -> tuple[tuple | None, str | None]:
        layout = affine_part(subject, plain=True)
        if layout is None or any(
            type(value) is not int
            for group in (layout.shape, layout.strides)
            for value in flatten(group)
        ):
            return None, f"{subject!r} is no static strided tensor a tensormap describes"
        modes = [
            (extent, step)
            for extents, steps in zip(layout.shape, layout.strides)
            for extent, step in zip(flatten(extents), flatten(steps))
            if extent > 1
        ]
        unit = [mode for mode in modes if mode[1] == 1]
        if len(unit) != 1:
            return None, (
                f"{len(unit)} of its modes step 1, and a tensormap's dim 0 is its one "
                "contiguous mode"
            )
        if len(modes) > len(self.steps) + 1:
            return None, (
                f"it is {len(modes)} modes, and a tensormap has at most "
                f"{len(self.steps) + 1} dims"
            )
        others = tuple(step for _, step in modes if step != 1)
        return others + (0,) * (len(self.steps) - len(others)), None

    def match(self, subject, captures=None):
        steps, _ = self.reading(subject)
        return None if steps is None else matched(SequencePattern(*self.steps), steps, captures)

    def refusal(self, subject, captures=None) -> str | None:
        held = dict(captures or {})
        steps, why = self.reading(subject)
        if steps is None:
            return why
        missed = _missed_place(self.steps, steps, held, first=1)
        if missed is not None:
            index, place, step = missed
            return (
                f"its dim {index} steps {step} elements, and a tensormap reads "
                f"{written_place(place.pattern, place.name)}"
            )
        return None

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        steps = written_tuple(("1", *(written_place(place) for place in self.steps)))
        return f"tensormap at {steps}"

    def relations(self) -> tuple[str, ...]:
        return (TENSORMAP_READING, *relations_of(self.steps))


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
    boxes = {TmaSwizzle.NONE: BoxPattern((_dim("dim0", dtype), *rest), dtype)}
    for index, mode in enumerate(
        (mode for mode in TmaSwizzle if mode.swizzle is not None),
        TMA_RANK,
    ):
        swizzle = mode.swizzle
        boxes[mode] = ComposedLayoutPattern(
            SwizzlePattern(swizzle.bits, swizzle.base, swizzle.shift),
            0,
            BoxPattern((_dim(f"dim{index}", dtype, span=mode.value), *rest), dtype, mode.value),
        )
    return BoxFamily(SWIZZLE_PLACE, boxes)


def TmaGlobalPattern(dtype: str) -> TensorMapPattern:
    return TensorMapPattern(
        tuple(
            CapturePattern(
                f"step{index}",
                BitsPattern(dtype, MultipleOfPattern(TMA_UNIT_BITS)),
            )
            for index in range(1, TMA_RANK)
        ),
        dtype,
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
    layout = LayoutPattern(((32,),), ((1,),), per_mode=True)
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
    "BOX_READING",
    "BoxFamily",
    "BoxPattern",
    "CopyAsyncTensor",
    "Run",
    "SWIZZLE_PLACE",
    "TENSORMAP_READING",
    "TMA_RANK",
    "TMA_STORAGES",
    "TMA_UNIT_BITS",
    "TensorMapPattern",
    "TmaBoxPattern",
    "TmaGlobalPattern",
    "TmaOperandPattern",
    "TmaSwizzle",
    "box_runs",
]
