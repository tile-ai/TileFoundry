"""Composable predicates for operation declarations and specialization dispatch.

The structural patterns, the ``Wildcard`` that names a captured value, and the
integer terms and formulas written over those names.
"""

from __future__ import annotations

import copy
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from itertools import count
from typing import Any, Callable, ClassVar, Iterable

from tilefoundry.ir import types, utils

from .match import UNREADABLE, Pattern
from .solve import evaluate

TOKENS = count()


@dataclass(frozen=True)
class Wildcard(Pattern):
    """Match any value, optionally binding it as an arithmetic variable."""

    name: str | None = None

    def term(self):
        """This wildcard as an integer term over its name."""
        return variable(self.name)

    @typing.override
    def captured(self) -> tuple[str, ...]:
        """The one name this wildcard binds, if it has a name."""
        return (self.name,) if self.name else ()

    def __add__(self, other):
        return self.term() + other

    def __radd__(self, other):
        return other + self.term()

    def __sub__(self, other):
        return self.term() - other

    def __rsub__(self, other):
        return other - self.term()

    def __mul__(self, other):
        return self.term() * other

    def __rmul__(self, other):
        return other * self.term()

    def __floordiv__(self, other):
        return self.term() // other

    def __rfloordiv__(self, other):
        return other // self.term()

    def __mod__(self, other):
        return self.term() % other

    def __rmod__(self, other):
        return other % self.term()

    def __eq__(self, other):
        if isinstance(other, Wildcard) and (not self.name or not other.name):
            return self.name == other.name
        return self.term() == other

    def __ne__(self, other):
        if isinstance(other, Wildcard) and (not self.name or not other.name):
            return self.name != other.name
        return self.term() != other

    def __lt__(self, other):
        return self.term() < other

    def __le__(self, other):
        return self.term() <= other

    def __gt__(self, other):
        return self.term() > other

    def __ge__(self, other):
        return self.term() >= other


@dataclass(frozen=True)
class Star(Pattern):
    """Match zero or more modes, applying one pattern to every mode."""

    pattern: Pattern
    repeats: ClassVar[bool] = True

    def __post_init__(self):
        if not isinstance(self.pattern, Pattern):
            raise TypeError("pattern.Star pattern must be a Pattern")


@dataclass(frozen=True, init=False)
class Or(Pattern):
    patterns: tuple

    def __init__(self, *patterns, predicates=()):
        object.__setattr__(self, "predicates", tuple(predicates))
        object.__setattr__(self, "patterns", tuple(patterns))


@dataclass(frozen=True)
class And(Pattern):
    parts: tuple = field(default_factory=tuple)


@dataclass(frozen=True, init=False)
class Sequence(Pattern):
    patterns: tuple

    def __init__(self, *patterns, predicates=()):
        object.__setattr__(self, "predicates", tuple(predicates))
        object.__setattr__(self, "patterns", tuple(patterns))


@dataclass(frozen=True)
class Range(Pattern):
    """A closed integer range, optionally naming a specialization dimension."""

    dim_var: str = ""
    lo: int | None = None
    hi: int | None = None

    def __post_init__(self):
        if not isinstance(self.dim_var, str):
            raise TypeError("pattern.Range dim_var must be a str")
        for name in ("lo", "hi"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise TypeError(
                    f"pattern.Range: {name} must be int or None, got {type(value).__name__}"
                )
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError("pattern.Range requires lo <= hi (closed [lo, hi])")
        if self.lo is None and self.hi is None:
            raise ValueError("pattern.Range must state a lower or upper bound")
        if self.dim_var and (self.lo is None or self.hi is None):
            raise ValueError("a named pattern.Range must state both lo and hi")


@dataclass(frozen=True, init=False)
class Switch(Pattern):
    param: str
    branches: tuple

    def __init__(self, param, branches, *, predicates=()):
        object.__setattr__(self, "predicates", tuple(predicates))
        object.__setattr__(self, "param", param)
        object.__setattr__(self, "branches", tuple(dict(branches).items()))


def mode_path(path: tuple[int, ...]) -> str:
    return "root" + "".join(f"[{index}]" for index in path)


def star_paths(value, path=()) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, Star):
        return (path,)
    if isinstance(value, tuple):
        return tuple(
            found for index, item in enumerate(value) for found in star_paths(item, (*path, index))
        )
    return ()


def validate_mode_frame(shape, strides, path=()) -> None:
    if isinstance(shape, tuple) or isinstance(strides, tuple):
        if not isinstance(shape, tuple) or not isinstance(strides, tuple):
            raise ValueError(
                f"pattern.Layout shape and strides must nest alike at {mode_path(path)}"
            )
        shape_stars = tuple(index for index, item in enumerate(shape) if isinstance(item, Star))
        stride_stars = tuple(index for index, item in enumerate(strides) if isinstance(item, Star))
        if len(shape_stars) > 1 or len(stride_stars) > 1:
            positions = shape_stars if len(shape_stars) > 1 else stride_stars
            raise ValueError(
                "pattern.Layout allows at most one pattern.Star per tuple at "
                f"{mode_path(path)}; found positions {positions}"
            )
        if len(shape) != len(strides):
            raise ValueError(
                f"pattern.Layout shape and strides must have the same arity at {mode_path(path)}"
            )
        for index, (extent, stride) in enumerate(zip(shape, strides)):
            validate_mode_frame(extent, stride, (*path, index))
        return
    if isinstance(shape, Star) != isinstance(strides, Star):
        raise ValueError(
            f"pattern.Layout shape and strides must place pattern.Star together at {mode_path(path)}"
        )


def binding_occurrences(value, depth=0, path=""):
    if isinstance(value, Star):
        yield from binding_occurrences(value.pattern, depth + 1, path)
        return
    if isinstance(value, Wildcard):
        name = getattr(value, "name", None)
        if name:
            yield name, depth, path
        return
    if isinstance(value, Pattern):
        for field_name, field_value in vars(value).items():
            yield from binding_occurrences(
                field_value,
                depth,
                f"{path}.{field_name}" if path else field_name,
            )
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from binding_occurrences(item, depth, f"{path}[{index}]")


def validate_binding_depths(shape, strides) -> None:
    occurrences: dict[str, list[tuple[int, str]]] = {}
    for field_name, value in (("shape", shape), ("strides", strides)):
        for name, depth, path in binding_occurrences(value, path=field_name):
            occurrences.setdefault(name, []).append((depth, path))
    for name, places in occurrences.items():
        if len({depth for depth, _ in places}) > 1:
            written = ", ".join(f"{path} (star depth {depth})" for depth, path in places)
            raise ValueError(
                f"pattern.Layout binding {name!r} appears at different star depths: {written}"
            )


@dataclass(frozen=True)
class Layout(Pattern):
    """An optional layout structure with named computed predicates."""

    shape: tuple | None = None
    strides: tuple | None = None

    def __post_init__(self):
        if self.shape is None or self.strides is None:
            field_name, paired = (
                ("shape", self.shape) if self.shape is not None else ("strides", self.strides)
            )
            stars = star_paths(paired)
            if stars:
                raise ValueError(
                    f"pattern.Layout {field_name} has pattern.Star at "
                    f"{mode_path(stars[0])}, but its paired field is absent"
                )
        else:
            validate_mode_frame(self.shape, self.strides)
        validate_binding_depths(self.shape, self.strides)

    @classmethod
    def from_layout(
        cls,
        layout,
        *,
        predicates=(),
    ):
        """Build the exact pattern for one authored arrangement."""
        held = tuple(predicates)
        if isinstance(layout, types.ComposedLayout):
            inner = layout.inner
            return ComposedLayout(
                Swizzle(inner.bits, inner.base, inner.shift)
                if isinstance(inner, types.Swizzle)
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
            *(types.flatten(self.shape) if self.shape is not None else ()),
            *(types.flatten(self.strides) if self.strides is not None else ()),
        )

    def fixed(self):
        if self.shape is None or self.strides is None:
            return None
        if any(isinstance(value, Pattern) for value in self.positions()):
            return None
        return types.Layout(tuple(self.shape), tuple(self.strides))


@dataclass(frozen=True)
class Swizzle(Pattern):
    bits: object
    base: object
    shift: object

    def fixed(self):
        if any(isinstance(value, Pattern) for value in (self.bits, self.base, self.shift)):
            return None
        return types.Swizzle(self.bits, self.base, self.shift)


@dataclass(frozen=True)
class ComposedLayout(Pattern):
    """Match composition fields, reading a bare layout as an identity composition."""

    inner: object = None
    offset: object = None
    outer: object = None

    def fixed(self):
        held = []
        for value in (self.inner, self.offset, self.outer):
            if isinstance(value, Pattern):
                if not hasattr(value, "fixed") or (value := value.fixed()) is None:
                    return None
            held.append(value)
        return None if any(value is None for value in held[1:]) else types.ComposedLayout(*held)


def require_per_mode(pattern) -> None:
    if isinstance(pattern, Or):
        for alternative in pattern.patterns:
            require_per_mode(alternative)
        return
    if isinstance(pattern, Star):
        require_per_mode(pattern.pattern)
        return
    if isinstance(pattern, ComposedLayout):
        pattern = pattern.outer
    if not isinstance(pattern, Layout) or any(
        getattr(predicate, "per_mode", True) is False for predicate in pattern.predicates
    ):
        raise ValueError(
            "pattern.Mesh layout predicates must check each top-level mode, "
            "or a pattern.ComposedLayout outer must use per-mode predicates"
        )


@dataclass(frozen=True)
class Mesh(Pattern):
    """Match named levels, or bind the innermost level with a wildcard."""

    topologies: tuple[str, ...] | Wildcard
    layout: object

    def __post_init__(self):
        if not isinstance(self.topologies, Wildcard) and (
            not self.topologies
            or any(not isinstance(name, str) or not name for name in self.topologies)
            or len(set(self.topologies)) != len(self.topologies)
        ):
            raise ValueError("pattern.Mesh topologies must be unique non-empty names")

        require_per_mode(self.layout)


@dataclass(frozen=True)
class Tensor(Pattern):
    """Match a ``TensorType`` of any rank, constrained by its stated fields."""

    dtype: Any = None
    shape: tuple | None = None
    storage: Any = None
    layout: Pattern | None = None


@dataclass(frozen=True)
class ShardLayout(Pattern):
    """Match a sharded layout's layout, shard attrs, and mesh frame."""

    layout: object
    attrs: tuple
    mesh: Mesh

    @typing.override
    def project(self, subject, bindings: dict):
        """Read a sharded subject through the participants this pattern's mesh names."""
        if not isinstance(subject, types.ShardLayout):
            return subject
        topologies = self.mesh.topologies
        if not isinstance(topologies, tuple):
            topologies = bindings.get(
                topologies.name,
                tuple(getattr(level, "name", level) for level in subject.mesh.topologies[-1:]),
            )
        return utils.participant_layout(subject, topologies)

    @typing.override
    def tensor_layout(self, subject, matcher):
        """Read a tensor's value states as shard attrs over its own mesh or the matcher's."""
        layout = subject.layout
        shard = layout if isinstance(layout, types.ShardLayout) else None
        mesh = shard.mesh if shard is not None else matcher.mesh
        if mesh is None or layout is None:
            return UNREADABLE
        if shard is None and isinstance(self.mesh.topologies, tuple):
            attrs = tuple(utils.attrs_by_axis(subject, mesh).values())
            mesh = utils.participant_layout(
                types.ShardLayout(layout, attrs, mesh), self.mesh.topologies
            ).mesh
        attrs = tuple(utils.attrs_by_axis(subject, mesh, matcher.owners).values())
        return types.ShardLayout(shard.layout if shard else layout, attrs, mesh)


def same_structure(left, right) -> bool:
    """Compare expression structure without invoking overloaded equality."""
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(
            same_structure(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    if isinstance(left, tuple):
        return len(left) == len(right) and all(same_structure(a, b) for a, b in zip(left, right))
    return left == right


def as_term(value) -> Term:
    if isinstance(value, Wildcard):
        return variable(value.name)
    return value if isinstance(value, Term) else Term("constant", (value,))


@dataclass(frozen=True, eq=False)
class Term:
    """An integer expression over named pattern bindings."""

    op: str
    args: tuple

    def replace(self, **changes):
        """A copy of this value with *changes* applied to its fields.

        ``dataclasses.replace`` cannot rebuild a value with ``init=False`` fields or with
        its own ``__init__``.
        """
        result = copy.copy(self)
        vars(result).update(changes)
        return result

    def binary(self, op: str, other) -> Term:
        return Term(op, (self, as_term(other)))

    def reflected(self, op: str, other) -> Term:
        return Term(op, (as_term(other), self))

    def __add__(self, other):
        return self.binary("add", other)

    def __radd__(self, other):
        return self.reflected("add", other)

    def __sub__(self, other):
        return self.binary("sub", other)

    def __rsub__(self, other):
        return self.reflected("sub", other)

    def __mul__(self, other):
        return self.binary("mul", other)

    def __rmul__(self, other):
        return self.reflected("mul", other)

    def __floordiv__(self, other):
        return self.binary("floordiv", other)

    def __rfloordiv__(self, other):
        return self.reflected("floordiv", other)

    def __mod__(self, other):
        return self.binary("mod", other)

    def __rmod__(self, other):
        return self.reflected("mod", other)

    def compare(self, op: str, other) -> Formula:
        return Formula(op, (self, as_term(other)))

    def __eq__(self, other):
        return self.compare("eq", other)

    def __ne__(self, other):
        return self.compare("ne", other)

    def __lt__(self, other):
        return self.compare("lt", other)

    def __le__(self, other):
        return self.compare("le", other)

    def __gt__(self, other):
        return self.compare("gt", other)

    def __ge__(self, other):
        return self.compare("ge", other)


def variable(name: str | None) -> Term:
    if not name:
        raise TypeError("an unnamed pattern.Wildcard cannot be used in a formula")
    return Term("variable", (name,))


@dataclass(frozen=True)
class Predicate(Pattern):
    """A computed condition over a subject and the matcher's bindings."""

    def holds(self, subject, bindings: dict) -> bool | None:
        """Return true, false, or None while required bindings are unknown."""
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class Formula(Predicate):
    """A Boolean expression evaluated against pattern bindings."""

    op: str
    args: tuple

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Formula)
            and self.op == other.op
            and same_structure(self.args, other.args)
        )

    def __ne__(self, other) -> bool:
        return not self == other

    def __and__(self, other):
        if not isinstance(other, Formula):
            return NotImplemented
        return Formula("and", (self, other))

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        if not isinstance(other, Formula):
            return NotImplemented
        return Formula("or", (self, other))

    def __ror__(self, other):
        return self.__or__(other)

    def __invert__(self):
        return Formula("not", (self,))

    def __bool__(self) -> bool:
        if self.op == "eq":
            return same_structure(*self.args)
        if self.op == "ne":
            return not same_structure(*self.args)
        raise TypeError(
            "Formula has no Python truth value; combine conditions with & / |, "
            "and use In(term, values) instead of Python 'in' or chained comparisons"
        )

    @typing.override
    def holds(self, subject, bindings: dict) -> bool | None:
        return evaluate(self, bindings)


def Bits(name: str) -> Term:
    """Read the bit width of the dtype bound to *name*."""
    return Term("bits", (name,))


@dataclass(frozen=True)
class Table:
    values: tuple

    def __init__(self, values: Iterable):
        object.__setattr__(self, "values", tuple(values))

    def __getitem__(self, index) -> Term:
        return Term("element", (self.values, as_term(index)))


def In(term, values: Iterable) -> Formula:
    return Formula("in", (as_term(term), tuple(values)))


def ForAll(term, build: Callable[[Term], Formula]) -> Formula:
    token = next(TOKENS)
    body = build(Term("bound", (token,)))
    if not isinstance(body, Formula):
        raise TypeError("ForAll body must produce a Formula")
    return Formula("forall", (as_term(term), token, body))


def Sum(term) -> Term:
    return Term("sum", (as_term(term),))


def Count(term) -> Term:
    return Term("count", (as_term(term),))


__all__ = [
    "And",
    "Bits",
    "ComposedLayout",
    "Count",
    "ForAll",
    "Formula",
    "In",
    "Layout",
    "Mesh",
    "Or",
    "Predicate",
    "Range",
    "Sequence",
    "ShardLayout",
    "Star",
    "Sum",
    "Swizzle",
    "Switch",
    "Table",
    "Tensor",
    "Term",
    "Wildcard",
    "same_structure",
    "variable",
]
