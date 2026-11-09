"""Build dimension arithmetic and decide whether a value is a dimension.

Dimension expressions are built without applying algebraic rules; a dimension is
normalized only when it enters IR. The ``simplify_dim`` service is implemented here.

See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
"""

from __future__ import annotations

from types import MappingProxyType

from tilefoundry import services
from tilefoundry.ir import expr, types

__all__ = [
    "KINDS",
    "i64_const",
    "KIND_OF",
    "as_dimension",
    "ceildiv",
    "combine",
    "is_dimension",
    "maximum",
    "minimum",
]

KINDS = MappingProxyType(
    {
        "add": expr.DimAdd,
        "sub": expr.DimSub,
        "mul": expr.DimMul,
        "floordiv": expr.DimFloorDiv,
        "mod": expr.DimMod,
        "min": expr.DimMin,
        "max": expr.DimMax,
    }
)


KIND_OF = MappingProxyType({cls: kind for kind, cls in KINDS.items()})


def wrap_operand(value):
    """A raw integer as an i64 ``Constant``; anything else unchanged."""
    if isinstance(value, int):
        return expr.Constant(type=types.TensorType.from_umat_scalar(), value=value)
    return value


def combine(kind: str, args: tuple) -> expr.Expr:
    """Build dimension arithmetic without applying algebraic rules.

    Raw integers become i64 constants and the dimension is normalized only when it
    enters IR. A bool operand raises, because falling back to ``NotImplemented``
    would let Python retry the operation as plain integer arithmetic and silently
    give the dimension the value 0 or 1. Operands outside
    ``ShapeDim = int | DimVar | Expr`` give ``NotImplemented``.
    """
    for operand in args:
        if isinstance(operand, bool):
            raise TypeError(
                f"dim arithmetic: bool operand {operand!r} is not a dimension; bool is an "
                "int subclass, so it is refused here rather than silently becoming 0 or 1"
            )
    if not all(isinstance(v, (int, expr.DimVar, expr.Expr)) for v in args):
        return NotImplemented
    ti64 = types.TensorType.from_umat_scalar()
    canon_args = tuple(wrap_operand(a) for a in args)
    return expr.Call(type=ti64, target=KINDS[kind](), args=canon_args)


def is_dimension(value) -> bool:
    """True iff *value* is a valid static-or-symbolic dim expression.

    A non-bool ``int``, a ``DimVar``, an ``int``-valued ``Constant``, or a
    ``Call`` over the dim-arithmetic ops whose args all satisfy this.
    """
    return dimension_in(value, {})


def dimension_in(value, known: dict[int, bool]) -> bool:
    """``is_dimension``, remembering the verdict of every node of one expression."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if not isinstance(value, (expr.DimVar, expr.Constant, expr.Call)):
        return False
    held = known.get(id(value))
    if held is not None:
        return held
    if isinstance(value, expr.DimVar):
        verdict = True
    elif isinstance(value, expr.Constant):
        verdict = isinstance(value.value, int) and not isinstance(value.value, bool)
    else:
        verdict = isinstance(value.target, expr.DIM_OP_TYPES) and all(
            isinstance(arg, (expr.DimVar, expr.Constant, expr.Call)) and dimension_in(arg, known)
            for arg in value.args
        )
    known[id(value)] = verdict
    return verdict


def as_dimension(value) -> expr.Expr:
    """Convert a dimension value to an expression without simplifying it."""
    if isinstance(value, expr.Expr):
        return value
    if isinstance(value, bool):
        raise TypeError("dim_expr: bool is not a dimension")
    if isinstance(value, int):
        return expr.Constant(type=types.TensorType.from_umat_scalar(), value=value)
    if isinstance(value, expr.DimVar):
        return combine("add", (value, 0))
    raise TypeError(f"dim_expr: expected int, DimVar, or Expr, got {type(value).__name__}")


def minimum(a, b) -> expr.Expr:
    """Build a symbolic ``min(a, b)`` dimension expression."""
    result = combine("min", (a, b))
    if result is NotImplemented:
        raise TypeError(
            f"dim_min: operands must be int, DimVar, or Expr, got "
            f"{type(a).__name__} and {type(b).__name__}"
        )
    return result


def maximum(a, b) -> expr.Expr:
    """Symbolic ``max(a, b)`` dim expression; see ``minimum``."""
    result = combine("max", (a, b))
    if result is NotImplemented:
        raise TypeError(
            f"dim_max: operands must be int, DimVar, or Expr, got "
            f"{type(a).__name__} and {type(b).__name__}"
        )
    return result


def ceildiv(a, b) -> expr.Expr:
    """Ceiling division ``(a + b - 1) // b`` as a dimension expression.

    Composes existing dim-arithmetic ops — there is no dedicated ceil-div
    op. Operands may be ``int`` (non-bool), ``DimVar`` or ``Expr``; the
    result is the same ``ShapeDim`` form produced by ``combine``.
    """
    num = combine("sub", (combine("add", (a, b)), 1))
    return combine("floordiv", (num, b))


services.registry.register(services.simplify_dim, combine)


def i64_const(value: int) -> expr.Constant:
    """The canonical i64 shape-scalar ``Constant`` (meta-scalar typed)."""
    return expr.Constant(type=types.TensorType.from_umat_scalar(), value=int(value))
