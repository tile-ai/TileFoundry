from __future__ import annotations

from dataclasses import dataclass, field, fields

from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.metadata import IRMetadata, diagnostic_location
from tilefoundry.ir.core.op import Op

from ..types.tensor_type import Type


@dataclass(frozen=True)
class Expr:
    """Typed SSA value. Base of all expression nodes (hir + tir-embedded).

    `type` is the Expr's result type (TensorType for single-output, TupleType
    for multi-output). It is kw-only so subclasses can declare positional
    fields without default-order clashes.
    """

    type: Type = field(kw_only=True)
    metadata: tuple[IRMetadata, ...] = field(
        default_factory=tuple,
        kw_only=True,
        compare=False,
        hash=False,
        repr=False,
    )

    def __post_init__(self) -> None:
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


def child_exprs(expr: Expr):
    """The Expr-valued children of *expr*, however deeply its fields nest them.

    Read off the node rather than listed per class, so a field holding a tuple
    of Functions or a tuple of (name, Function) pairs is reached like any other.
    """
    def walk(value):
        if isinstance(value, Expr):
            yield value
        elif isinstance(value, tuple):
            for item in value:
                yield from walk(item)

    for member in fields(expr):
        yield from walk(getattr(expr, member.name, None))


@dataclass(frozen=True)
class Var(Expr):
    name: str
    is_const: bool = False


@dataclass(frozen=True)
class Constant(Expr):
    value: object


@dataclass(frozen=True)
class Call(Expr):
    """Call to an Op. Produces a value. Cannot be top-level Stmt in tir."""

    target: Op
    args: tuple[Expr, ...]

    def _dim_binop(self, other, op_name: str, *, reverse: bool = False):
        from tilefoundry.ir.types import dim  # noqa: PLC0415

        if not dim.is_dim_op_call(self):
            return NotImplemented
        operands = (other, self) if reverse else (self, other)
        return dim._dim_binop(getattr(dim, op_name), *operands)

    def __add__(self, other):
        return self._dim_binop(other, "DimAdd")

    def __radd__(self, other):
        return self._dim_binop(other, "DimAdd", reverse=True)

    def __sub__(self, other):
        return self._dim_binop(other, "DimSub")

    def __rsub__(self, other):
        return self._dim_binop(other, "DimSub", reverse=True)

    def __mul__(self, other):
        return self._dim_binop(other, "DimMul")

    def __rmul__(self, other):
        return self._dim_binop(other, "DimMul", reverse=True)

    def __floordiv__(self, other):
        return self._dim_binop(other, "DimFloorDiv")

    def __rfloordiv__(self, other):
        return self._dim_binop(other, "DimFloorDiv", reverse=True)

    def __mod__(self, other):
        return self._dim_binop(other, "DimMod")

    def __rmod__(self, other):
        return self._dim_binop(other, "DimMod", reverse=True)


@dataclass(frozen=True)
class Tuple(Expr):
    """Value-form explicit tuple construction.

    ``Tuple((a, b))``: the type is ``TupleType(fields=(a.type, b.type))``. Not
    a registered Op — an IR-level construct emitted by the parser for
    ``return (a, b)`` bodies and per-axis scalar tuples (e.g. ``insert_slice``
    offsets).
    """

    elements: tuple[Expr, ...]
