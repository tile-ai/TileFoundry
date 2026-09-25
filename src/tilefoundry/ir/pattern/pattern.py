"""Composable predicates for operation declarations and specialization dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from tilefoundry.ir.types import (
    Broadcast,
    ComposedLayout,
    Layout,
    Mesh,
    ShardLayout,
    Swizzle,
    TensorType,
    make_mesh,
)
from tilefoundry.ir.types.int_tuple import congruent
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.layout_algebra import (
    ASYNC_WIDTHS,
    box_runs,
    frame_of,
    is_inverse_projectable,
)
from tilefoundry.ir.types.mesh import separate

from .constraint import affine_part
from .match import (
    ABSENT,
    ARRANGEMENT,
    UNNAMED_PLACE,
    Match,
    _named,
    alternatives_of,
    evaluated,
    matched,
    relations_of,
    resolved,
    written_alternatives,
    written_binding,
    written_grouped,
    written_place,
    written_tuple,
)


@dataclass(frozen=True)
class Pattern:
    """Base class for a predicate that returns bindings or ``None``."""

    def match(self, subject, captures=None) -> Match | None:
        raise NotImplementedError

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return type(self).__name__

    def relations(self) -> tuple[str, ...]:
        return ()

    def alternatives(self, bindings=()) -> tuple:
        return ((tuple(bindings), self),)

    def rules(self, arrangements=None) -> tuple[str, ...]:
        return self.relations()

    def resolve(self, bindings):
        return replace(
            self,
            **{name: resolved(value, bindings) for name, value in vars(self).items()},
        )


@dataclass(frozen=True)
class WildcardPattern(Pattern):
    """Match any value without binding it."""

    def match(self, subject, captures=None) -> Match:
        return Match(dict(captures or {}))

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return name


VECTOR_READING = (
    "every run: each tile axis's modes walked fastest first, contiguous ones joined; "
    "the run at step 1 and every other step a whole number of vectors"
)


def vector_widths(layout, element_bits: int) -> tuple[int, ...]:
    """Every cp.async width that divides every run in an arrangement."""
    widest = ASYNC_WIDTHS[-1]
    if isinstance(layout, ShardLayout):
        layout = layout.layout
    inner = getattr(layout, "inner", None)
    if inner is not None and hasattr(inner, "base"):
        widest = min(widest, 1 << inner.base)
    held = affine_part(layout)
    if held is None or any(
        type(value) is not int
        for group in (held.shape, held.strides)
        for value in flatten(group)
    ):
        return ()
    runs = box_runs(held, element_bits, None, limit=None)
    unit = [run.extent for run in runs if run.step == 1]
    if len(unit) != 1:
        return ()
    counted = (unit[0], *(run.step for run in runs if run.step != 1))
    return tuple(
        width
        for width in ASYNC_WIDTHS
        if width <= widest
        and all(value * element_bits % (width * 8) == 0 for value in counted)
    )


@dataclass(frozen=True)
class VectorPattern(Pattern):
    """An arrangement that moves whole 4-, 8-, or 16-byte vectors."""

    width: CapturePattern
    dtype: str

    def widths(self, subject, captures) -> tuple[int, ...]:
        bits = getattr(dict(captures or {}).get(self.dtype), "bit_width", None)
        return () if type(bits) is not int else vector_widths(subject, bits)

    def match(self, subject, captures=None):
        held = dict(captures or {})
        widths = self.widths(subject, held)
        if not widths:
            return None
        if self.width.name in held:
            return Match(held) if held[self.width.name] in widths else None
        return matched(self.width, widths[-1], held)

    def refusal(self, subject, captures=None) -> str | None:
        if self.match(subject, captures) is not None:
            return None
        sizes = ", ".join(map(str, ASYNC_WIDTHS[:-1])) + f" or {ASYNC_WIDTHS[-1]}"
        return (
            f"{subject!r} moves no whole vector of {sizes} bytes -- its run at step 1 "
            "and every other step are no whole number of one -- so the two ends share "
            "no run wide enough for cp.async"
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return f"vectors of {self.width.name} bytes"

    def relations(self) -> tuple[str, ...]:
        return (VECTOR_READING, *relations_of((self.width,)))


@dataclass(frozen=True, init=False)
class OrPattern(Pattern):
    patterns: tuple

    def __init__(self, *patterns):
        object.__setattr__(self, "patterns", tuple(patterns))

    def match(self, subject, captures=None):
        for pattern in self.patterns:
            held = matched(pattern, subject, captures)
            if held is not None:
                return held
        return None

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        if not any(isinstance(pattern, Pattern) for pattern in self.patterns):
            values = "{" + ", ".join(written_place(p) for p in self.patterns) + "}"
            return values if name == UNNAMED_PLACE else f"{name} in {values}"
        return written_alternatives(self.alternatives(), name)

    def relations(self) -> tuple[str, ...]:
        return relations_of(self.patterns)

    def alternatives(self, bindings=()) -> tuple:
        return tuple(
            held for pattern in self.patterns for held in alternatives_of(pattern, bindings)
        )

    def resolve(self, bindings):
        held = tuple(
            pattern
            for pattern in (resolved(one, bindings) for one in self.patterns)
            if pattern is not ABSENT
        )
        return OrPattern(*held) if held else ABSENT


@dataclass(frozen=True)
class AndPattern(Pattern):
    parts: tuple = field(default_factory=tuple)

    def match(self, subject, captures=None):
        held = Match(dict(captures or {}))
        for pattern in self.parts:
            held = matched(pattern, subject, held.captures)
            if held is None:
                return None
        return held

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return " and ".join(written_place(pattern, name) for pattern in self.parts)

    def relations(self) -> tuple[str, ...]:
        return relations_of(self.parts)


@dataclass(frozen=True, init=False)
class SequencePattern(Pattern):
    patterns: tuple

    def __init__(self, *patterns):
        object.__setattr__(self, "patterns", tuple(patterns))

    def match(self, subject, captures=None):
        if not isinstance(subject, (tuple, list)) or len(subject) != len(self.patterns):
            return None
        held = Match(dict(captures or {}))
        for pattern, value in zip(self.patterns, subject):
            held = matched(pattern, value, held.captures)
            if held is None:
                return None
        return held

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_tuple(tuple(written_place(p, name) for p in self.patterns))

    def relations(self) -> tuple[str, ...]:
        return relations_of(self.patterns)

    def resolve(self, bindings):
        return SequencePattern(*(resolved(pattern, bindings) for pattern in self.patterns))


@dataclass(frozen=True)
class CapturePattern(Pattern):
    name: str
    pattern: object = None

    def match(self, subject, captures=None):
        held = dict(captures or {})
        if self.name in held:
            return Match(held) if held[self.name] == subject else None
        found = matched(self.pattern, subject, held)
        if found is None:
            return None
        return Match({**found.captures, self.name: subject})

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return self.name

    def relations(self) -> tuple[str, ...]:
        if self.pattern is None:
            return ()
        return (written_place(self.pattern, self.name), *relations_of((self.pattern,)))


@dataclass(frozen=True, init=False)
class ConstraintPattern(Pattern):
    patterns: tuple

    def __init__(self, *patterns):
        if not patterns:
            raise ValueError("a constraint pattern must state at least one constraint")
        object.__setattr__(self, "patterns", tuple(patterns))

    def match(self, subject, captures=None):
        held = Match(dict(captures or {}))
        for pattern in self.patterns:
            held = matched(pattern, subject, held.captures)
            if held is None:
                return None
        return held

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return " and ".join(written_place(pattern, name) for pattern in self.patterns)

    def resolve(self, bindings):
        return self


@dataclass(frozen=True)
class MultipleOfPattern(Pattern):
    unit: int

    def __post_init__(self):
        if type(self.unit) is not int or self.unit <= 0:
            raise ValueError("MultipleOfPattern unit must be a positive int")

    def match(self, subject, captures=None):
        return (
            Match(dict(captures or {}))
            if type(subject) is int and subject % self.unit == 0
            else None
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return f"{name} % {self.unit} = 0"


@dataclass(frozen=True)
class RangePattern(Pattern):
    """A closed integer range, optionally naming a specialization dimension."""

    dim_var: str = ""
    lo: int | None = None
    hi: int | None = None

    def __post_init__(self):
        if not isinstance(self.dim_var, str):
            raise TypeError("RangePattern dim_var must be a str")
        for name in ("lo", "hi"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise TypeError(
                    f"RangePattern: {name} must be int or None, got {type(value).__name__}"
                )
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError("RangePattern requires lo <= hi (closed [lo, hi])")
        if self.lo is None and self.hi is None:
            raise ValueError("RangePattern must state a lower or upper bound")
        if self.dim_var and (self.lo is None or self.hi is None):
            raise ValueError("a named RangePattern must state both lo and hi")

    def match(self, subject, captures=None):
        if isinstance(subject, bool) or not isinstance(subject, int):
            return None
        if self.lo is not None and subject < self.lo:
            return None
        if self.hi is not None and subject > self.hi:
            return None
        return Match(dict(captures or {}))

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        if self.lo is None:
            return f"{name} <= {self.hi}"
        if self.hi is None:
            return f"{self.lo} <= {name}"
        return f"{self.lo} <= {name} <= {self.hi}"


@dataclass(frozen=True)
class OneOfPattern(Pattern):
    values: tuple

    def __post_init__(self):
        if len(self.values) < 2:
            raise ValueError("OneOfPattern requires at least two values")

    def match(self, subject, captures=None):
        return (
            Match(dict(captures or {})) if any(subject == value for value in self.values) else None
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return f"{name} in {{{', '.join(_named(value) for value in self.values)}}}"


@dataclass(frozen=True)
class AttrPattern(Pattern):
    attr: str
    pattern: object

    def match(self, subject, captures=None):
        if not hasattr(subject, self.attr):
            return None
        return matched(self.pattern, getattr(subject, self.attr), captures)

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_place(self.pattern, f"{name}.{self.attr}")


@dataclass(frozen=True)
class BitsPattern(Pattern):
    dtype: str
    pattern: object

    def match(self, subject, captures=None):
        width = getattr(dict(captures or {}).get(self.dtype), "bit_width", None)
        if type(subject) is not int or type(width) is not int:
            return None
        return matched(self.pattern, subject * width, captures)

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_place(self.pattern, f"{name} * {self.dtype}.bit_width")


@dataclass(frozen=True, init=False)
class SwitchPattern(Pattern):
    param: str
    branches: tuple

    def __init__(self, param, branches):
        object.__setattr__(self, "param", param)
        object.__setattr__(self, "branches", tuple(dict(branches).items()))

    def match(self, subject, captures=None):
        held = dict(captures or {})
        if self.param in held:
            wanted = held[self.param]
            pattern = next((p for value, p in self.branches if value == wanted), None)
            return None if pattern is None else matched(pattern, subject, held)
        for value, pattern in self.branches:
            found = matched(pattern, subject, {**held, self.param: value})
            if found is not None:
                return found
        return None

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_alternatives(self.alternatives(), name)

    def relations(self) -> tuple[str, ...]:
        return relations_of(tuple(pattern for _, pattern in self.branches))

    def refusal(self, subject, captures=None) -> str | None:
        held = dict(captures or {})
        if self.param not in held:
            return (
                None
                if self.match(subject, held) is not None
                else f"{subject!r} is none of {len(self.branches)} branches"
            )
        pattern = next((p for value, p in self.branches if value == held[self.param]), None)
        if pattern is None:
            return f"{self.param}={written_binding(held[self.param])} selects no branch"
        explained = getattr(pattern, "refusal", None)
        if explained is not None:
            return explained(subject, held)
        return (
            None
            if matched(pattern, subject, held) is not None
            else f"{subject!r} is not {written_place(pattern)}"
        )

    def alternatives(self, bindings=()) -> tuple:
        return tuple(
            held
            for value, pattern in self.branches
            for held in alternatives_of(pattern, (*bindings, (self.param, value)))
        )

    def resolve(self, bindings):
        held = dict(bindings or {})
        if self.param in held:
            pattern = next((p for value, p in self.branches if value == held[self.param]), ABSENT)
            return ABSENT if pattern is ABSENT else resolved(pattern, held)
        branches = {value: resolved(pattern, held) for value, pattern in self.branches}
        branches = {value: p for value, p in branches.items() if p is not ABSENT}
        return SwitchPattern(self.param, branches) if branches else ABSENT


@dataclass(frozen=True)
class GuardPattern(Pattern):
    symbol: object
    condition: Pattern
    pattern: object

    def match(self, subject, captures=None):
        value = evaluated(self.symbol, captures)
        if value is None or matched(self.condition, value, captures) is None:
            return None
        return matched(self.pattern, subject, captures)

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_place(self.pattern, name)

    def relations(self) -> tuple[str, ...]:
        return (
            *relations_of((self.pattern,)),
            self.condition.describe(str(self.symbol)),
        )

    def resolve(self, bindings):
        value = evaluated(self.symbol, bindings)
        if value is None:
            return GuardPattern(self.symbol, self.condition, resolved(self.pattern, bindings))
        return (
            resolved(self.pattern, bindings)
            if matched(self.condition, value) is not None
            else ABSENT
        )

    def fixed(self):
        return None


@dataclass(frozen=True)
class LayoutPattern(Pattern):
    """An affine layout, preserving grouping and optional whole-layout rules."""

    shape: tuple
    strides: tuple
    forward: bool = True
    injective: bool = True
    per_mode: bool = False

    def positions(self) -> tuple:
        return (*flatten(self.shape), *flatten(self.strides))

    def match(self, subject, captures=None):
        layout = subject
        if not isinstance(layout, Layout) or layout.strides is None:
            return None
        if not (
            congruent(layout.shape, self.shape)
            and congruent(layout.strides, self.strides)
        ):
            return None
        extents = tuple(flatten(layout.shape))
        strides = tuple(flatten(layout.strides))
        if any(type(number) is not int for number in (*extents, *strides)):
            return None
        if any(number <= 0 for number in extents):
            return None
        held = matched(SequencePattern(*self.positions()), (*extents, *strides), captures)
        if held is None:
            return None
        arrangements = (
            tuple(
                Layout(tuple(flatten(shape)), tuple(flatten(steps)))
                for shape, steps in zip(layout.shape, layout.strides)
            )
            if self.per_mode
            else (Layout(extents, strides),)
        )
        for arrangement in arrangements:
            held_strides = tuple(flatten(arrangement.strides))
            if self.forward and any(stride < 0 for stride in held_strides):
                return None
            if self.injective and not is_inverse_projectable(arrangement):
                return None
        return held

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return (
            f"Layout({written_grouped(tuple(self.shape))}, {written_grouped(tuple(self.strides))})"
        )

    def relations(self) -> tuple[str, ...]:
        lines = list(relations_of(self.positions()))
        subject = "each top-level mode" if self.per_mode else ARRANGEMENT
        if self.forward:
            lines.append(f"{subject} has no backward step")
        if self.injective:
            lines.append(f"{subject} reaches each of its own slots exactly once")
        return tuple(lines)

    def fixed(self):
        if any(isinstance(value, Pattern) for value in self.positions()):
            return None
        return Layout(tuple(self.shape), tuple(self.strides))


@dataclass(frozen=True)
class SwizzlePattern(Pattern):
    bits: int
    base: int
    shift: int

    def match(self, subject, captures=None):
        return (
            Match(dict(captures or {}))
            if subject == Swizzle(self.bits, self.base, self.shift)
            else None
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return f"Swizzle({self.bits}, {self.base}, {self.shift})"

    def fixed(self):
        return Swizzle(self.bits, self.base, self.shift)


@dataclass(frozen=True)
class ComposedLayoutPattern(Pattern):
    """Match the three fields of a concrete ``ComposedLayout`` only."""

    inner: object = None
    offset: object = None
    outer: object = None

    def match(self, subject, captures=None):
        if not isinstance(subject, ComposedLayout):
            return None
        held = Match(dict(captures or {}))
        for pattern, value in (
            (self.inner, subject.inner),
            (self.offset, subject.offset),
            (self.outer, subject.outer),
        ):
            held = matched(pattern, value, held.captures)
            if held is None:
                return None
        return held

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return (
            f"ComposedLayout({written_place(self.inner)}, "
            f"{written_place(self.offset)}, {written_place(self.outer)})"
        )

    def relations(self) -> tuple[str, ...]:
        return relations_of((self.inner, self.offset, self.outer))

    def fixed(self):
        held = tuple(
            value.fixed() if isinstance(value, Pattern) and hasattr(value, "fixed") else value
            for value in (self.inner, self.offset, self.outer)
        )
        return None if any(value is None for value in held[1:]) else ComposedLayout(*held)


@dataclass(frozen=True)
class MeshPattern(Pattern):
    """Match named mesh levels after separating and recomposing them.

    A ``ComposedLayoutPattern`` here deliberately matches only a sliced mesh,
    because an unsliced mesh carries a bare ``Layout``. To accept both forms,
    use ``OrPattern(ComposedLayoutPattern(offset=..., outer=L), L)``.
    """

    topologies: tuple[str, ...]
    layout: object

    def __post_init__(self):
        if (
            not self.topologies
            or any(not isinstance(name, str) or not name for name in self.topologies)
            or len(set(self.topologies)) != len(self.topologies)
        ):
            raise ValueError("MeshPattern topologies must be unique non-empty names")

        def require_per_mode(pattern) -> None:
            if isinstance(pattern, OrPattern):
                for alternative in pattern.patterns:
                    require_per_mode(alternative)
                return
            if isinstance(pattern, ComposedLayoutPattern):
                pattern = pattern.outer
            if not isinstance(pattern, LayoutPattern) or not pattern.per_mode:
                raise ValueError(
                    "MeshPattern layout must be a LayoutPattern(per_mode=True), "
                    "or a ComposedLayoutPattern whose outer uses per_mode=True"
                )

        require_per_mode(self.layout)

    def match(self, subject, captures=None):
        if not isinstance(subject, Mesh):
            return None
        picked = tuple(
            level
            for level in separate(subject)
            if getattr(level.topologies[0], "name", level.topologies[0]) in self.topologies
        )
        found = tuple(getattr(level.topologies[0], "name", level.topologies[0]) for level in picked)
        if len(picked) != len(self.topologies) or set(found) != set(self.topologies):
            return None
        return matched(self.layout, make_mesh(*picked).layout, captures)

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return f"Mesh({self.topologies!r}, {written_place(self.layout)})"

    def relations(self) -> tuple[str, ...]:
        return relations_of((self.layout,))


@dataclass(frozen=True)
class ScalarPattern(Pattern):
    def match(self, subject, captures=None):
        return (
            Match(dict(captures or {}))
            if isinstance(subject, TensorType) and subject.shape == ()
            else None
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return "scalar"


@dataclass(frozen=True)
class TensorPattern(Pattern):
    """Match a non-scalar ``TensorType`` field by field."""

    rank: int | None = None
    dtype: Any = None
    shape: tuple | None = None
    storage: Any = None
    layout: Pattern | None = None

    def match(self, subject, captures=None):
        if not isinstance(subject, TensorType) or subject.shape == ():
            return None
        wanted = (
            (self.rank, len(subject.shape)),
            (self.dtype, subject.dtype),
            (self.storage, subject.storage),
        )
        if self.shape is not None:
            wanted = ((SequencePattern(*self.shape), tuple(subject.shape)), *wanted)
        if self.layout is not None:
            wanted = (*wanted, (self.layout, subject.layout))
        held = Match(dict(captures or {}))
        for pattern, value in wanted:
            held = matched(pattern, value, held.captures)
            if held is None:
                return None
        return held

    def refusal(self, subject, captures=None) -> str | None:
        if not isinstance(subject, TensorType) or subject.shape == ():
            return f"{subject!r} is no tensor"
        wanted = (
            (
                "shape",
                None if self.shape is None else SequencePattern(*self.shape),
                tuple(subject.shape),
            ),
            ("rank", self.rank, len(subject.shape)),
            ("dtype", self.dtype, subject.dtype),
            ("storage", self.storage, subject.storage),
            ("layout", self.layout, subject.layout),
        )
        held = Match(dict(captures or {}))
        for name, pattern, value in wanted:
            if pattern is None:
                continue
            found = matched(pattern, value, held.captures)
            if found is not None:
                held = found
                continue
            explained = getattr(pattern, "refusal", None)
            if name == "layout" and explained is not None:
                return explained(value, held.captures)
            return f"its {name} is {_named(value) if name != 'shape' else value}"
        return None

    def describe(self, name: str = UNNAMED_PLACE, arrangements=None) -> str:
        stated = []
        if self.shape is not None:
            stated.append("shape=" + written_tuple(tuple(written_place(x) for x in self.shape)))
        if self.rank is not None:
            stated.append(f"rank={self.rank}")
        if self.dtype is not None:
            stated.append(f"dtype={_named(self.dtype)}")
        if self.storage is not None:
            stated.append(f"storage={_named(self.storage)}")
        head = " ".join(stated) if stated else "any tensor"
        return (
            f"{head}, in any arrangement"
            if self.layout is None
            else (f"{head}, held in {self.layout.describe()}")
        )

    def relations(self) -> tuple[str, ...]:
        return relations_of(
            (
                self.rank,
                self.dtype,
                self.storage,
                *(self.shape or ()),
                *((self.layout,) if self.layout is not None else ()),
            )
        )


@dataclass(frozen=True)
class ShardLayoutPattern(Pattern):
    """Match a sharded layout's arrangement, shard attrs, and mesh frame."""

    arrangement: object
    attrs: tuple
    mesh: Mesh

    def match(self, subject, captures=None):
        if not isinstance(subject, ShardLayout):
            return None
        extra = len(subject.attrs) - len(self.attrs)
        if (
            extra < 0
            or subject.attrs[extra:] != self.attrs
            or any(not isinstance(attr, Broadcast) for attr in subject.attrs[:extra])
        ):
            return None
        framed = frame_of(subject.mesh.layout)
        if framed is None:
            return None
        frame = framed[1]
        if extra:
            if not isinstance(frame, Layout):
                return None
            frame = Layout(frame.shape[extra:], frame.strides[extra:])
        subject_names = tuple(
            getattr(topology, "name", topology) for topology in subject.mesh.topologies
        )
        wanted_names = tuple(
            getattr(topology, "name", topology) for topology in self.mesh.topologies
        )
        if frame != self.mesh.layout or subject_names != wanted_names:
            return None
        return self.reads(subject.layout, captures)

    def reads(self, layout, captures=None):
        return matched(self.arrangement, layout, captures)

    def accepts_layout(self, layout) -> bool:
        return self.reads(layout) is not None

    def alternatives(self, bindings=()) -> tuple:
        return alternatives_of(self.arrangement, bindings)

    def relations(self) -> tuple[str, ...]:
        return relations_of((self.arrangement,))

    def rules(self, arrangements=None) -> tuple[str, ...]:
        items = self.alternatives() if arrangements is None else tuple(arrangements)
        return relations_of(tuple(pattern for _, pattern in items))

    def describe(self, name: str = UNNAMED_PLACE, arrangements=None) -> str:
        items = self.alternatives() if arrangements is None else tuple(arrangements)
        head = f"{len(items)} arrangement{'' if len(items) == 1 else 's'}:"
        written = written_alternatives(items).splitlines()
        return "\n".join(
            (
                head,
                *(f"  {line}" for line in written),
                *(f"  {rule}" for rule in self.rules(items)),
            )
        )


Scalar: ScalarPattern = ScalarPattern()
Tensor: TensorPattern = TensorPattern()


__all__ = [
    "AndPattern",
    "AttrPattern",
    "BitsPattern",
    "CapturePattern",
    "ComposedLayoutPattern",
    "ConstraintPattern",
    "GuardPattern",
    "LayoutPattern",
    "MeshPattern",
    "MultipleOfPattern",
    "OneOfPattern",
    "OrPattern",
    "Pattern",
    "RangePattern",
    "Scalar",
    "ScalarPattern",
    "SequencePattern",
    "ShardLayoutPattern",
    "SwizzlePattern",
    "SwitchPattern",
    "Tensor",
    "TensorPattern",
    "VectorPattern",
    "WildcardPattern",
    "vector_widths",
]
