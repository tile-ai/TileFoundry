"""The typed SSA value every IR expression is, and the nodes that need no Op.

``Expr`` is the base of every expression node of HIR and TIR. ``Call`` applies
an Op; its arithmetic operators build dimension expressions through the
``simplify_dim`` service, which returns ``NotImplemented`` for operands that are
not dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from tilefoundry import services

from .errors import VerifyError
from .metadata import Annotated, IRMetadata, diagnostic_location
from .op import Op

__all__ = ["Call", "Constant", "Expr", "Tuple", "Var", "child_exprs"]


@dataclass(unsafe_hash=True)
class Expr(Annotated):
    """Typed SSA value. Base of all expression nodes (hir + tir-embedded).

    `type` is the Expr's result type (TensorType for single-output, TupleType
    for multi-output). It is kw-only so subclasses can declare positional
    fields without default-order clashes.
    """

    type: Type = field(kw_only=True, hash=False)
    metadata: tuple[IRMetadata, ...] = field(
        default_factory=tuple,
        kw_only=True,
        compare=False,
        hash=False,
        repr=False,
    )

    def retype(self, type: Type) -> None:
        """Replace the carried type, keeping this value's identity."""
        self.type = type

    def __post_init__(self) -> None:
        """Reject metadata that is not an IRMetadata or repeats a concrete class."""
        seen: set[type[IRMetadata]] = set()
        for value in self.metadata:
            if not isinstance(value, IRMetadata):
                location = diagnostic_location(self)
                where = f"\n  at {location}" if location else ""
                raise VerifyError(
                    f"{type(self).__name__} metadata entries must be IRMetadata, "
                    f"got {type(value).__name__}{where}"
                )
            value_cls = type(value)
            if value_cls in seen:
                location = diagnostic_location(self)
                where = f"\n  at {location}" if location else ""
                raise VerifyError(
                    f"{type(self).__name__} has duplicate {value_cls.__name__} metadata{where}"
                )
            seen.add(value_cls)


def expr_values(value):
    """The Expr values inside *value*, however deeply tuples nest them."""
    if isinstance(value, Expr):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from expr_values(item)


def child_exprs(expr: Expr):
    """The Expr-valued children of *expr*, however deeply its fields nest them.

    Read off the node rather than listed per class, so a field holding a tuple
    of Functions or a tuple of (name, Function) pairs is reached like any other.
    """
    for member in fields(expr):
        yield from expr_values(getattr(expr, member.name, None))


@dataclass(unsafe_hash=True)
class Var(Expr):
    """A named value: a parameter, a binding or an induction variable."""

    name: str
    is_const: bool = False

    annotation: Type | None = field(default=None, kw_only=True, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Default the annotation to the value's own type."""
        super().__post_init__()
        if self.annotation is None:
            self.annotation = self.type

    def restamp(self, annotation: Type | None, type: Type) -> None:
        """Replace the declared annotation and the carried type, keeping this Var's identity."""
        self.annotation = annotation
        self.retype(type)


@dataclass(unsafe_hash=True)
class Constant(Expr):
    """A literal value."""

    value: object


@dataclass(unsafe_hash=True)
class Call(Expr):
    """Call to an Op. Produces a value. Cannot be top-level Stmt in tir."""

    target: Op
    args: tuple[Expr, ...]

    def __add__(self, other):
        """Dimension sum."""
        return call_binop(self, other, "add", reverse=False)

    def __radd__(self, other):
        """Dimension sum with *other* on the left."""
        return call_binop(self, other, "add", reverse=True)

    def __sub__(self, other):
        """Dimension difference."""
        return call_binop(self, other, "sub", reverse=False)

    def __rsub__(self, other):
        """Dimension difference with *other* on the left."""
        return call_binop(self, other, "sub", reverse=True)

    def __mul__(self, other):
        """Dimension product."""
        return call_binop(self, other, "mul", reverse=False)

    def __rmul__(self, other):
        """Dimension product with *other* on the left."""
        return call_binop(self, other, "mul", reverse=True)

    def __floordiv__(self, other):
        """Dimension floor quotient."""
        return call_binop(self, other, "floordiv", reverse=False)

    def __rfloordiv__(self, other):
        """Dimension floor quotient with *other* on the left."""
        return call_binop(self, other, "floordiv", reverse=True)

    def __mod__(self, other):
        """Dimension remainder."""
        return call_binop(self, other, "mod", reverse=False)

    def __rmod__(self, other):
        """Dimension remainder with *other* on the left."""
        return call_binop(self, other, "mod", reverse=True)


def call_binop(call: Call, other, kind: str, *, reverse: bool):
    """The dimension expression *kind* of *call* and *other*, or ``NotImplemented``.

    Only a call over a dimension operation is a dimension; the operation says so
    through its ``dimension_arithmetic`` marker.
    """
    if not getattr(type(call.target), "dimension_arithmetic", False):
        return NotImplemented
    operands = (other, call) if reverse else (call, other)
    return services.simplify_dim(kind, operands)


@dataclass(unsafe_hash=True)
class Tuple(Expr):
    """Value-form explicit tuple construction.

    ``Tuple((a, b))``: the type is ``TupleType(fields=(a.type, b.type))``. Not
    a registered Op — an IR-level construct emitted by the parser for
    ``return (a, b)`` bodies and per-axis scalar tuples (e.g. ``insert_slice``
    offsets).
    """

    elements: tuple[Expr, ...]
