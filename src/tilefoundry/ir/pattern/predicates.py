"""Named computed predicates for operation-declaration layout patterns."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.ir.types.layout_algebra import coalesce, is_inverse_projectable

from .match import (
    ARRANGEMENT,
    UNNAMED_PLACE,
    Match,
    matched,
    relations_of,
    written_place,
    written_tuple,
)
from .pattern import CapturePattern, Predicate, SequencePattern

VECTOR_READING = (
    "every run: each tile axis's modes walked fastest first, contiguous ones joined; "
    "the run at step 1 and every other step a whole number of vectors"
)
TENSORMAP_READING = "every tensormap: one dim per mode of the tile, the mode at step 1 first"


def _arrangements(layout: Layout, per_mode: bool) -> tuple[Layout, ...]:
    if per_mode:
        return tuple(
            Layout(tuple(flatten(shape)), tuple(flatten(steps)))
            for shape, steps in zip(layout.shape, layout.strides)
        )
    return (Layout(tuple(flatten(layout.shape)), tuple(flatten(layout.strides))),)


@dataclass(frozen=True)
class Forward(Predicate):
    """Require nonnegative steps, across the whole layout or per top-level mode."""

    per_mode: bool = False

    def holds(self, arrangement: Layout, captures: dict) -> bool:
        return all(
            all(step >= 0 for step in flatten(part.strides))
            for part in _arrangements(arrangement, self.per_mode)
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        subject = "each top-level mode" if self.per_mode else ARRANGEMENT
        return f"{subject} with no backward step"

    def relations(self) -> tuple[str, ...]:
        subject = "each top-level mode" if self.per_mode else ARRANGEMENT
        return (f"{subject} has no backward step",)


@dataclass(frozen=True)
class Injective(Predicate):
    """Require every slot to be reached once, across the layout or per mode."""

    per_mode: bool = False

    def holds(self, arrangement: Layout, captures: dict) -> bool:
        return all(
            is_inverse_projectable(part) for part in _arrangements(arrangement, self.per_mode)
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        subject = "each top-level mode" if self.per_mode else ARRANGEMENT
        return f"{subject} reaching each of its own slots exactly once"

    def relations(self) -> tuple[str, ...]:
        subject = "each top-level mode" if self.per_mode else ARRANGEMENT
        return (f"{subject} reaches each of its own slots exactly once",)


def _reverse_group(group):
    if isinstance(group, tuple):
        return tuple(_reverse_group(mode) for mode in reversed(group))
    return group


def _row_major_groups_for_cute(layout: Layout) -> Layout:
    """Reverse modes within each tile axis for CuTe's mode-0-fast algebra."""
    return Layout(
        shape=tuple(_reverse_group(group) for group in layout.shape),
        strides=tuple(_reverse_group(group) for group in layout.strides),
    )


def _vector_widths(layout, element_bits: int, widths: tuple[int, ...]) -> tuple[int, ...]:
    """Every requested byte width that divides every run in an arrangement."""
    if not widths:
        return ()
    widest = widths[-1]
    stated = layout.layout if isinstance(layout, ShardLayout) else layout
    inner = getattr(stated, "inner", None)
    if inner is not None and hasattr(inner, "base"):
        widest = min(widest, 1 << inner.base)
    held = Predicate.arrangement(layout)
    if held is None:
        return ()
    grouped = coalesce(
        _row_major_groups_for_cute(held),
        (0,) * len(held.shape),
    )
    runs = tuple(
        sorted(
            zip(flatten(grouped.shape), flatten(grouped.strides)),
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

    width: CapturePattern
    dtype: str
    widths: tuple[int, ...]

    def available_widths(self, subject, captures) -> tuple[int, ...]:
        bits = getattr(dict(captures or {}).get(self.dtype), "bit_width", None)
        layout = subject.layout if isinstance(subject, ShardLayout) else subject
        return () if type(bits) is not int else _vector_widths(layout, bits, self.widths)

    def match(self, subject, captures=None):
        held = dict(captures or {})
        widths = self.available_widths(subject, held)
        if not widths:
            return None
        if self.width.name in held:
            return Match(held) if held[self.width.name] in widths else None
        return matched(self.width, widths[-1], held)

    def refusal(self, subject, captures=None) -> str | None:
        if self.match(subject, captures) is not None:
            return None
        sizes = " or ".join(map(str, self.widths))
        if len(self.widths) > 2:
            sizes = ", ".join(map(str, self.widths[:-1])) + f" or {self.widths[-1]}"
        return (
            f"{subject!r} moves no whole vector of {sizes} bytes -- its run at step 1 "
            "and every other step are no whole number of one -- so the two ends share "
            "no run wide enough for the requested vector widths"
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return f"vectors of {self.width.name} bytes"

    def relations(self) -> tuple[str, ...]:
        return (VECTOR_READING, *relations_of((self.width,)))


@dataclass(frozen=True)
class PlainArrangement(Predicate):
    """Require an arrangement with no transform and no nonzero offset."""

    @staticmethod
    def _stated(subject):
        return subject.layout if isinstance(subject, ShardLayout) else subject

    def holds(self, arrangement: Layout, captures: dict) -> bool:
        return True

    def match(self, subject, captures=None):
        stated = self._stated(subject)
        if isinstance(stated, ComposedLayout) and (stated.inner is not None or stated.offset != 0):
            return None
        return super().match(subject, captures)

    def refusal(self, subject, captures=None) -> str | None:
        if self.match(subject, captures) is not None:
            return None
        stated = self._stated(subject)
        if isinstance(stated, ComposedLayout):
            reached = []
            if stated.inner is not None:
                reached.append(f"through {stated.inner!r}")
            if stated.offset != 0:
                reached.append(f"at offset {stated.offset}")
            if reached:
                return f"it is reached {' '.join(reached)}, not as a plain arrangement"
        return f"{subject!r} is no static strided arrangement"

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return "a plain arrangement with no transform or offset"

    def relations(self) -> tuple[str, ...]:
        return ("every plain arrangement has no transform or nonzero offset",)


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
    limit: int,
) -> tuple[Run, ...]:
    """Read box runs by tile axis, ordered by increasing step."""
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
                and joined <= limit
                and not (span is not None and last.step == 1 and joined * element_bits > span * 8)
            ):
                runs[-1] = replace(last, extent=joined)
            else:
                runs.append(Run(extent, step, axis, mode))
    return tuple(sorted(runs, key=lambda run: run.step))


def _missed_place(places, values, captures, first: int = 0) -> tuple | None:
    held = captures
    for index, (place, value) in enumerate(zip(places, values), first):
        found = matched(place, value, held)
        if found is None:
            return index, place, value
        held = found.captures
    return None


@dataclass(frozen=True)
class BoxDims(Predicate):
    """Require one box dimension per contiguous run of tile-axis modes."""

    dims: tuple
    dtype: str
    limit: int
    span: int | None = None

    def reading(self, subject, captures) -> tuple[tuple | None, str | None]:
        layout = self.arrangement(subject)
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
        reading = (
            "every box: each tile axis's modes, contiguous ones joined up to "
            f"{self.limit} elements, one dim each, in increasing step"
        )
        return (reading, *relations_of(self.dims))


@dataclass(frozen=True)
class TensorMap(Predicate):
    """Require one tensor-map dimension per nontrivial tile mode."""

    steps: tuple
    dtype: str

    def reading(self, subject) -> tuple[tuple | None, str | None]:
        layout = self.arrangement(subject)
        if layout is None:
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
                f"it is {len(modes)} modes, and a tensormap has at most {len(self.steps) + 1} dims"
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


__all__ = [
    "BoxDims",
    "Forward",
    "Injective",
    "PlainArrangement",
    "TensorMap",
    "WholeVectors",
]
