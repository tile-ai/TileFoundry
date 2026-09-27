"""Composable predicates for operation declarations and specialization dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tilefoundry.ir.types import (
    Broadcast,
    ComposedLayout,
    Layout,
    ShardLayout,
    Swizzle,
)
from tilefoundry.ir.types.layout import flatten

from .match import (
    UNNAMED_PLACE,
    Match,
    PatternMatcher,
    _named,
    alternatives_of,
    matched,
    relations_of,
    written_alternatives,
    written_grouped,
    written_place,
    written_tuple,
)


@dataclass(frozen=True)
class Pattern:
    """Base class for a predicate that returns bindings or ``None``."""

    def match(self, subject, captures=None) -> Match | None:
        held = PatternMatcher(dict(captures or {}))
        return Match(held.bindings) if held.match(self, subject) and held.solve() else None

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return type(self).__name__

    def relations(self) -> tuple[str, ...]:
        return ()

    def rules(self, arrangements=None) -> tuple[str, ...]:
        return self.relations()


@dataclass(frozen=True)
class Predicate(Pattern):
    """A computed condition over a subject and the matcher's bindings."""

    @staticmethod
    def arrangement(subject) -> Layout | None:
        """Read the static strided layout beneath shard and composition wrappers."""
        if isinstance(subject, ShardLayout):
            if not all(isinstance(attr, Broadcast) for attr in subject.attrs):
                return None
            subject = subject.layout
        if isinstance(subject, ComposedLayout):
            if subject.inner is not None and not isinstance(subject.inner, Swizzle):
                return None
            subject = subject.outer
        if not isinstance(subject, Layout) or subject.strides is None:
            return None
        extents = tuple(flatten(subject.shape))
        strides = tuple(flatten(subject.strides))
        if any(type(number) is not int for number in (*extents, *strides)):
            return None
        if any(extent <= 0 for extent in extents):
            return None
        return subject

    def holds(self, subject, bindings: dict) -> bool | None:
        """Return true, false, or None while required bindings are unknown."""
        raise NotImplementedError

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        raise NotImplementedError

    def relations(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class WildcardPattern(Pattern):
    """Match any value without binding it."""

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return name


@dataclass(frozen=True, init=False)
class OrPattern(Pattern):
    patterns: tuple

    def __init__(self, *patterns):
        object.__setattr__(self, "patterns", tuple(patterns))

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        if not any(isinstance(pattern, Pattern) for pattern in self.patterns):
            values = "{" + ", ".join(written_place(p) for p in self.patterns) + "}"
            return values if name == UNNAMED_PLACE else f"{name} in {values}"
        return written_alternatives(alternatives_of(self), name)

    def relations(self) -> tuple[str, ...]:
        return relations_of(self.patterns)


@dataclass(frozen=True)
class AndPattern(Pattern):
    parts: tuple = field(default_factory=tuple)

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return " and ".join(written_place(pattern, name) for pattern in self.parts)

    def relations(self) -> tuple[str, ...]:
        return relations_of(self.parts)


@dataclass(frozen=True, init=False)
class SequencePattern(Pattern):
    patterns: tuple

    def __init__(self, *patterns):
        object.__setattr__(self, "patterns", tuple(patterns))

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_tuple(tuple(written_place(p, name) for p in self.patterns))

    def relations(self) -> tuple[str, ...]:
        return relations_of(self.patterns)


@dataclass(frozen=True)
class CapturePattern(Pattern):
    name: str
    pattern: object = None

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

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return " and ".join(written_place(pattern, name) for pattern in self.patterns)


@dataclass(frozen=True)
class MultipleOfPattern(Pattern):
    unit: int

    def __post_init__(self):
        if type(self.unit) is not int or self.unit <= 0:
            raise ValueError("MultipleOfPattern unit must be a positive int")

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

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return f"{name} in {{{', '.join(_named(value) for value in self.values)}}}"


@dataclass(frozen=True)
class AttrPattern(Pattern):
    attr: str
    pattern: object

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_place(self.pattern, f"{name}.{self.attr}")


@dataclass(frozen=True)
class BitsPattern(Pattern):
    dtype: str
    pattern: object

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_place(self.pattern, f"{name} * {self.dtype}.bit_width")


@dataclass(frozen=True, init=False)
class SwitchPattern(Pattern):
    param: str
    branches: tuple

    def __init__(self, param, branches):
        object.__setattr__(self, "param", param)
        object.__setattr__(self, "branches", tuple(dict(branches).items()))

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_alternatives(alternatives_of(self), name)

    def relations(self) -> tuple[str, ...]:
        return relations_of(tuple(pattern for _, pattern in self.branches))


@dataclass(frozen=True)
class GuardPattern(Pattern):
    symbol: object
    condition: Pattern
    pattern: object

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return written_place(self.pattern, name)

    def relations(self) -> tuple[str, ...]:
        return (
            *relations_of((self.pattern,)),
            self.condition.describe(str(self.symbol)),
        )

    def fixed(self):
        return None


@dataclass(frozen=True)
class LayoutPattern(Pattern):
    """An optional layout structure with named computed predicates."""

    shape: tuple | None = None
    strides: tuple | None = None
    predicates: tuple[Predicate, ...] = field(default_factory=tuple)

    @classmethod
    def from_layout(
        cls,
        layout,
        *,
        predicates: tuple[Predicate, ...] = (),
    ):
        """Build the exact pattern for one authored arrangement."""
        held = tuple(predicates)
        if isinstance(layout, ComposedLayout):
            inner = layout.inner
            return ComposedLayoutPattern(
                SwizzlePattern(inner.bits, inner.base, inner.shift)
                if isinstance(inner, Swizzle)
                else inner,
                layout.offset,
                cls.from_layout(
                    layout.outer,
                    predicates=held,
                ),
            )
        return cls(
            tuple(layout.shape),
            tuple(layout.strides),
            predicates=held,
        )

    def positions(self) -> tuple:
        return (
            *(flatten(self.shape) if self.shape is not None else ()),
            *(flatten(self.strides) if self.strides is not None else ()),
        )

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        if self.shape is None and self.strides is None:
            return "layout"
        shape = name if self.shape is None else written_grouped(tuple(self.shape))
        strides = name if self.strides is None else written_grouped(tuple(self.strides))
        return f"Layout({shape}, {strides})"

    def relations(self) -> tuple[str, ...]:
        return (*relations_of(self.positions()), *relations_of(self.predicates))

    def fixed(self):
        if self.shape is None or self.strides is None:
            return None
        if any(isinstance(value, Pattern) for value in self.positions()):
            return None
        return Layout(tuple(self.shape), tuple(self.strides))


@dataclass(frozen=True)
class SwizzlePattern(Pattern):
    bits: int
    base: int
    shift: int

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
            if not isinstance(pattern, LayoutPattern) or any(
                getattr(predicate, "per_mode", True) is False for predicate in pattern.predicates
            ):
                raise ValueError(
                    "MeshPattern layout predicates must check each top-level mode, "
                    "or a ComposedLayoutPattern outer must use per-mode predicates"
                )

        require_per_mode(self.layout)

    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return f"Mesh({self.topologies!r}, {written_place(self.layout)})"

    def relations(self) -> tuple[str, ...]:
        return relations_of((self.layout,))


@dataclass(frozen=True)
class ScalarPattern(Pattern):
    def describe(self, name: str = UNNAMED_PLACE) -> str:
        return "scalar"


@dataclass(frozen=True)
class TensorPattern(Pattern):
    """Match a non-scalar ``TensorType`` field by field."""

    dtype: Any = None
    shape: tuple | None = None
    storage: Any = None
    layout: Pattern | None = None

    def describe(self, name: str = UNNAMED_PLACE, arrangements=None) -> str:
        stated = []
        if self.shape is not None:
            stated.append("shape=" + written_tuple(tuple(written_place(x) for x in self.shape)))
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
                self.dtype,
                self.storage,
                *(self.shape or ()),
                *((self.layout,) if self.layout is not None else ()),
            )
        )


@dataclass(frozen=True)
class ShardLayoutPattern(Pattern):
    """Match a sharded layout's layout, shard attrs, and mesh frame."""

    layout: object
    attrs: tuple
    mesh: MeshPattern

    def reads(self, layout, captures=None):
        return matched(self.layout, layout, captures)

    def accepts_layout(self, layout) -> bool:
        return self.reads(layout) is not None

    def relations(self) -> tuple[str, ...]:
        return relations_of((self.layout,))

    def rules(self, arrangements=None) -> tuple[str, ...]:
        items = alternatives_of(self) if arrangements is None else tuple(arrangements)
        return relations_of(tuple(pattern for _, pattern in items))

    def describe(self, name: str = UNNAMED_PLACE, arrangements=None) -> str:
        items = alternatives_of(self) if arrangements is None else tuple(arrangements)
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
    "Predicate",
    "RangePattern",
    "Scalar",
    "ScalarPattern",
    "SequencePattern",
    "ShardLayoutPattern",
    "SwizzlePattern",
    "SwitchPattern",
    "Tensor",
    "TensorPattern",
    "WildcardPattern",
]
