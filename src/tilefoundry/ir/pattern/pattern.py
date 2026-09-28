"""Composable predicates for operation declarations and specialization dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from tilefoundry.ir.types import ComposedLayout, Layout, Swizzle
from tilefoundry.ir.types.layout import flatten

from .match import Match, PatternMatcher


@dataclass(frozen=True)
class Pattern:
    """Base class for a predicate that returns bindings or ``None``."""

    def match(self, subject, captures=None) -> Match | None:
        held = PatternMatcher(dict(captures or {}))
        return Match(dict(held.bindings)) if held.match(self, subject) and held.solve() else None

@dataclass(frozen=True)
class WildcardPattern(Pattern):
    """Match any value, optionally binding it as an arithmetic variable."""

    name: str | None = None

    def _term(self):
        from .predicates import variable  # noqa: PLC0415 - expression protocol cycle

        return variable(self.name)

    def __add__(self, other):
        return self._term() + other

    def __radd__(self, other):
        return other + self._term()

    def __sub__(self, other):
        return self._term() - other

    def __rsub__(self, other):
        return other - self._term()

    def __mul__(self, other):
        return self._term() * other

    def __rmul__(self, other):
        return other * self._term()

    def __floordiv__(self, other):
        return self._term() // other

    def __rfloordiv__(self, other):
        return other // self._term()

    def __mod__(self, other):
        return self._term() % other

    def __rmod__(self, other):
        return other % self._term()

    def __eq__(self, other):
        if isinstance(other, WildcardPattern) and (not self.name or not other.name):
            return self.name == other.name
        return self._term() == other

    def __ne__(self, other):
        if isinstance(other, WildcardPattern) and (not self.name or not other.name):
            return self.name != other.name
        return self._term() != other

    def __lt__(self, other):
        return self._term() < other

    def __le__(self, other):
        return self._term() <= other

    def __gt__(self, other):
        return self._term() > other

    def __ge__(self, other):
        return self._term() >= other

@dataclass(frozen=True)
class StarPattern(Pattern):
    """Match zero or more modes, applying one pattern to every mode."""

    pattern: Pattern

    def __post_init__(self):
        if not isinstance(self.pattern, Pattern):
            raise TypeError("StarPattern pattern must be a Pattern")

@dataclass(frozen=True, init=False)
class OrPattern(Pattern):
    patterns: tuple

    def __init__(self, *patterns):
        object.__setattr__(self, "patterns", tuple(patterns))

@dataclass(frozen=True)
class AndPattern(Pattern):
    parts: tuple = field(default_factory=tuple)


@dataclass(frozen=True, init=False)
class SequencePattern(Pattern):
    patterns: tuple

    def __init__(self, *patterns):
        object.__setattr__(self, "patterns", tuple(patterns))

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

@dataclass(frozen=True, init=False)
class SwitchPattern(Pattern):
    param: str
    branches: tuple

    def __init__(self, param, branches):
        object.__setattr__(self, "param", param)
        object.__setattr__(self, "branches", tuple(dict(branches).items()))

def _mode_path(path: tuple[int, ...]) -> str:
    return "root" + "".join(f"[{index}]" for index in path)


def _star_paths(value, path=()) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, StarPattern):
        return (path,)
    if isinstance(value, tuple):
        return tuple(
            found
            for index, item in enumerate(value)
            for found in _star_paths(item, (*path, index))
        )
    return ()


def _validate_mode_frame(shape, strides, path=()) -> None:
    if isinstance(shape, tuple) or isinstance(strides, tuple):
        if not isinstance(shape, tuple) or not isinstance(strides, tuple):
            raise ValueError(
                "LayoutPattern shape and strides must nest alike at "
                f"{_mode_path(path)}"
            )
        shape_stars = tuple(
            index for index, item in enumerate(shape) if isinstance(item, StarPattern)
        )
        stride_stars = tuple(
            index for index, item in enumerate(strides) if isinstance(item, StarPattern)
        )
        if len(shape_stars) > 1 or len(stride_stars) > 1:
            positions = shape_stars if len(shape_stars) > 1 else stride_stars
            raise ValueError(
                "LayoutPattern allows at most one StarPattern per tuple at "
                f"{_mode_path(path)}; found positions {positions}"
            )
        if len(shape) != len(strides):
            raise ValueError(
                "LayoutPattern shape and strides must have the same arity at "
                f"{_mode_path(path)}"
            )
        for index, (extent, stride) in enumerate(zip(shape, strides)):
            _validate_mode_frame(extent, stride, (*path, index))
        return
    if isinstance(shape, StarPattern) != isinstance(strides, StarPattern):
        raise ValueError(
            "LayoutPattern shape and strides must place StarPattern together at "
            f"{_mode_path(path)}"
        )


def _binding_occurrences(value, depth=0, path=""):
    if isinstance(value, StarPattern):
        yield from _binding_occurrences(value.pattern, depth + 1, path)
        return
    if isinstance(value, WildcardPattern):
        name = getattr(value, "name", None)
        if name:
            yield name, depth, path
        return
    if isinstance(value, Pattern):
        for field_name, field_value in vars(value).items():
            yield from _binding_occurrences(
                field_value,
                depth,
                f"{path}.{field_name}" if path else field_name,
            )
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _binding_occurrences(item, depth, f"{path}[{index}]")


def _validate_binding_depths(shape, strides) -> None:
    occurrences: dict[str, list[tuple[int, str]]] = {}
    for field_name, value in (("shape", shape), ("strides", strides)):
        for name, depth, path in _binding_occurrences(value, path=field_name):
            occurrences.setdefault(name, []).append((depth, path))
    for name, places in occurrences.items():
        if len({depth for depth, _ in places}) > 1:
            written = ", ".join(f"{path} (star depth {depth})" for depth, path in places)
            raise ValueError(
                f"LayoutPattern binding {name!r} appears at different star depths: {written}"
            )


@dataclass(frozen=True)
class LayoutPattern(Pattern):
    """An optional layout structure with named computed predicates."""

    shape: tuple | None = None
    strides: tuple | None = None
    predicates: tuple[Predicate, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.shape is None or self.strides is None:
            field_name, paired = (
                ("shape", self.shape) if self.shape is not None else ("strides", self.strides)
            )
            stars = _star_paths(paired)
            if stars:
                raise ValueError(
                    f"LayoutPattern {field_name} has StarPattern at "
                    f"{_mode_path(stars[0])}, but its paired field is absent"
                )
        else:
            _validate_mode_frame(self.shape, self.strides)
        _validate_binding_depths(self.shape, self.strides)

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

    def fixed(self):
        if self.shape is None or self.strides is None:
            return None
        if any(isinstance(value, Pattern) for value in self.positions()):
            return None
        return Layout(tuple(self.shape), tuple(self.strides))


@dataclass(frozen=True)
class SwizzlePattern(Pattern):
    bits: object
    base: object
    shift: object

    def fixed(self):
        if any(isinstance(value, Pattern) for value in (self.bits, self.base, self.shift)):
            return None
        return Swizzle(self.bits, self.base, self.shift)


@dataclass(frozen=True)
class ComposedLayoutPattern(Pattern):
    """Match composition fields, reading a bare layout as an identity composition."""

    inner: object = None
    offset: object = None
    outer: object = None
    predicates: tuple[Predicate, ...] = field(default_factory=tuple)
    issues_per_row: Callable[[dict], tuple[int, int] | None] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def fixed(self):
        held = []
        for value in (self.inner, self.offset, self.outer):
            if isinstance(value, Pattern):
                if not hasattr(value, "fixed") or (value := value.fixed()) is None:
                    return None
            held.append(value)
        return None if any(value is None for value in held[1:]) else ComposedLayout(*held)


@dataclass(frozen=True)
class MeshPattern(Pattern):
    """Match named mesh levels after separating and recomposing them."""

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
            if isinstance(pattern, StarPattern):
                require_per_mode(pattern.pattern)
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

@dataclass(frozen=True)
class ScalarPattern(Pattern):
    pass


@dataclass(frozen=True)
class TensorPattern(Pattern):
    """Match a non-scalar ``TensorType`` field by field."""

    dtype: Any = None
    shape: tuple | None = None
    storage: Any = None
    layout: Pattern | None = None
    predicates: tuple[Predicate, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class ShardLayoutPattern(Pattern):
    """Match a sharded layout's layout, shard attrs, and mesh frame."""

    layout: object
    attrs: tuple
    mesh: MeshPattern
    predicates: tuple[Predicate, ...] = field(default_factory=tuple)

Scalar: ScalarPattern = ScalarPattern()
Tensor: TensorPattern = TensorPattern()


__all__ = [
    "AndPattern",
    "ComposedLayoutPattern",
    "LayoutPattern",
    "MeshPattern",
    "OrPattern",
    "Pattern",
    "RangePattern",
    "Scalar",
    "ScalarPattern",
    "SequencePattern",
    "ShardLayoutPattern",
    "StarPattern",
    "SwizzlePattern",
    "SwitchPattern",
    "Tensor",
    "TensorPattern",
    "WildcardPattern",
]
