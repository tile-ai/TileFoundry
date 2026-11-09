"""Named computed predicates for operation-declaration layout patterns."""

from __future__ import annotations

import typing
from dataclasses import dataclass, replace

from tilefoundry.ir import types
from tilefoundry.ir.types import layout_algebra

from .constraint import strided_arrangement
from .match import matched
from .pattern import Predicate, Sequence, Tensor, Wildcard


def arrangement(subject, *, allow_partial=False) -> types.Layout | None:
    """Read the static strided layout beneath shard and composition wrappers."""
    subject = strided_arrangement(subject, allow_partial=allow_partial)
    if subject is None:
        return None
    extents = tuple(types.flatten(subject.shape))
    strides = tuple(types.flatten(subject.strides))
    if any(type(number) is not int for number in (*extents, *strides)):
        return None
    if any(extent <= 0 for extent in extents):
        return None
    return subject


@dataclass(frozen=True)
class Ranked(Predicate):
    """Require a tensor with at least one logical axis."""

    @typing.override
    def holds(self, subject, bindings: dict) -> bool:
        return isinstance(subject, types.TensorType) and len(subject.shape) > 0


def is_scalar_tensor(**fields) -> Tensor:
    """Construct a rank-zero tensor pattern with optional field constraints."""
    return Tensor(shape=(), **fields)


def is_ranked_tensor(**fields) -> Tensor:
    """Construct a tensor pattern that requires at least one logical axis."""
    return Tensor(predicates=(Ranked(),), **fields)


def arrangements(layout: types.Layout, per_mode: bool) -> tuple[types.Layout, ...]:
    if per_mode:
        return tuple(
            types.Layout(tuple(types.flatten(shape)), tuple(types.flatten(steps)))
            for shape, steps in zip(layout.shape, layout.strides)
        )
    return (types.Layout(tuple(types.flatten(layout.shape)), tuple(types.flatten(layout.strides))),)


@dataclass(frozen=True)
class Forward(Predicate):
    """Require nonnegative steps, across the whole layout or per top-level mode."""

    per_mode: bool = False

    @typing.override
    def holds(self, subject, bindings: dict) -> bool:
        held = arrangement(subject)
        if held is None:
            return False
        return all(
            all(step >= 0 for step in types.flatten(part.strides))
            for part in arrangements(held, self.per_mode)
        )


@dataclass(frozen=True)
class Injective(Predicate):
    """Require every slot to be reached once, across the layout or per mode."""

    per_mode: bool = False

    @typing.override
    def holds(self, subject, bindings: dict) -> bool:
        held = arrangement(subject)
        if held is None:
            return False
        return all(
            layout_algebra.is_inverse_projectable(part)
            for part in arrangements(held, self.per_mode)
        )


def reverse_group(group):
    if isinstance(group, tuple):
        return tuple(reverse_group(mode) for mode in reversed(group))
    return group


def row_major_groups_for_cute(layout: types.Layout) -> types.Layout:
    """Reverse modes within each tile axis for CuTe's mode-0-fast algebra."""
    return types.Layout(
        shape=tuple(reverse_group(group) for group in layout.shape),
        strides=tuple(reverse_group(group) for group in layout.strides),
    )


def vector_widths(layout, element_bits: int, widths: tuple[int, ...]) -> tuple[int, ...]:
    """Every requested byte width that divides every run in an arrangement."""
    if not widths:
        return ()
    widest = widths[-1]
    stated = layout.layout if isinstance(layout, types.ShardLayout) else layout
    inner = getattr(stated, "inner", None)
    if inner is not None and hasattr(inner, "base"):
        widest = min(widest, 1 << inner.base)
    held = arrangement(layout)
    if held is None:
        return ()
    grouped = layout_algebra.coalesce(
        row_major_groups_for_cute(held),
        (0,) * len(held.shape),
    )
    runs = tuple(
        sorted(
            zip(types.flatten(grouped.shape), types.flatten(grouped.strides)),
            key=lambda run: run[1],
        )
    )
    unit = [extent for extent, step in runs if step == 1]
    if len(unit) != 1:
        return ()
    counted = (unit[0], *(step for _, step in runs if step != 1))
    return tuple(
        width
        for width in widths
        if width <= widest and all(value * element_bits % (width * 8) == 0 for value in counted)
    )


@dataclass(frozen=True)
class WholeVectors(Predicate):
    """Require whole vectors at one of the requested byte widths."""

    width: Wildcard
    dtype: str
    widths: tuple[int, ...]

    def available_widths(self, subject, captures) -> tuple[int, ...]:
        bits = getattr(dict(captures or {}).get(self.dtype), "bit_width", None)
        layout = subject.layout if isinstance(subject, types.ShardLayout) else subject
        return () if type(bits) is not int else vector_widths(layout, bits, self.widths)

    @typing.override
    def holds(self, subject, bindings: dict) -> bool:
        widths = self.available_widths(subject, bindings)
        if not widths:
            return False
        if self.width.name in bindings:
            return bindings[self.width.name] in widths
        return matched(self.width, widths[-1], bindings) is not None


@dataclass(frozen=True)
class PlainArrangement(Predicate):
    """Require an arrangement with no transform and no nonzero offset."""

    allow_partial: bool = False

    @typing.override
    def holds(self, subject, bindings: dict) -> bool:
        stated = subject.layout if isinstance(subject, types.ShardLayout) else subject
        if isinstance(stated, types.ComposedLayout) and (
            stated.inner is not None or stated.offset != 0
        ):
            return False
        return arrangement(subject, allow_partial=self.allow_partial) is not None


@dataclass(frozen=True)
class Run:
    """One contiguous run of modes from one logical tile axis."""

    extent: int
    step: int
    axis: int
    mode: int


def box_runs(
    layout: types.Layout,
    element_bits: int,
    span: int | None,
    limit: int,
) -> tuple[Run, ...]:
    """Read box runs by tile axis, ordered by increasing step."""
    runs: list[Run] = []
    for axis, (extents, steps) in enumerate(zip(layout.shape, layout.strides)):
        modes = tuple(enumerate(zip(types.flatten(extents), types.flatten(steps))))
        for mode, (extent, step) in reversed(modes):
            if extent == 1:
                continue
            last = runs[-1] if runs and runs[-1].axis == axis else None
            joined = None if last is None else last.extent * extent
            if (
                last is not None
                and step == last.step * last.extent
                and joined <= limit
                and not (span is not None and last.step == 1 and joined * element_bits > span * 8)
            ):
                runs[-1] = replace(last, extent=joined)
            else:
                runs.append(Run(extent, step, axis, mode))
    return tuple(sorted(runs, key=lambda run: run.step))


@dataclass(frozen=True)
class BoxDims(Predicate):
    """Require one box dimension per contiguous run of tile-axis modes."""

    dims: tuple
    dtype: str
    limit: int
    span: int | None = None
    allow_partial: bool = False

    def reading(self, subject, captures) -> tuple[tuple | None, str | None]:
        layout = arrangement(subject, allow_partial=self.allow_partial)
        if layout is None:
            return None, f"{subject!r} is no static strided arrangement"
        width = getattr(captures.get(self.dtype), "bit_width", None)
        if type(width) is not int:
            return None, f"the element it arranges is not bound as {self.dtype}"
        runs = box_runs(layout, width, self.span, self.limit)
        if not runs:
            return None, "it holds one element, which is no box"
        if len(runs) > len(self.dims):
            return None, (
                f"it is {len(runs)} runs of modes, and a box has at most {len(self.dims)} dims"
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

    @typing.override
    def holds(self, subject, bindings: dict) -> bool:
        extents, unlaid = self.reading(subject, bindings)
        return (
            extents is not None
            and unlaid is None
            and matched(Sequence(*self.dims), extents, bindings) is not None
        )


@dataclass(frozen=True)
class TensorMap(Predicate):
    """Require one tensor-map dimension per nontrivial tile mode."""

    steps: tuple
    dtype: str

    def reading(self, subject) -> tuple[tuple | None, str | None]:
        layout = arrangement(subject)
        if layout is None:
            return None, f"{subject!r} is no static strided tensor a tensormap describes"
        modes = [
            (extent, step)
            for extents, steps in zip(layout.shape, layout.strides)
            for extent, step in zip(types.flatten(extents), types.flatten(steps))
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
                f"it is {len(modes)} modes, and a tensormap has at most {len(self.steps) + 1} dims"
            )
        others = tuple(step for _, step in modes if step != 1)
        return others + (0,) * (len(self.steps) - len(others)), None

    @typing.override
    def holds(self, subject, bindings: dict) -> bool:
        steps, _ = self.reading(subject)
        return steps is not None and matched(Sequence(*self.steps), steps, bindings) is not None


__all__ = [
    "BoxDims",
    "Forward",
    "Injective",
    "PlainArrangement",
    "Ranked",
    "TensorMap",
    "WholeVectors",
    "arrangement",
    "is_ranked_tensor",
    "is_scalar_tensor",
]
