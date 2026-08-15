from __future__ import annotations

from ..core.expr import Call, Constant, Expr
from ..core.op import Op
from ..core.param_def import ParamDef


class DimConst(Op):
    value = ParamDef(kind="attribute", annotation=int)


class _DimVarMeta(type(Op)):
    def __call__(cls, name=None, lo=None, hi=None, **attrs):

        if name is None:
            name = attrs.pop("name", None)
        if lo is None:
            lo = attrs.pop("lo", None)
        if hi is None:
            hi = attrs.pop("hi", None)
        if name is None or not isinstance(name, str) or not name:
            raise TypeError("DimVar requires a non-empty str name=")
        if not isinstance(lo, int) or isinstance(lo, bool):
            raise TypeError(f"DimVar({name!r}): lo must be int, got {type(lo).__name__}")
        if not isinstance(hi, int) or isinstance(hi, bool):
            raise TypeError(f"DimVar({name!r}): hi must be int, got {type(hi).__name__}")
        if not (lo < hi):
            raise ValueError(
                f"DimVar({name!r}, {lo}, {hi}): require lo < hi "
                f"(half-open envelope [lo, hi); a fixed dim is [k, k+1))"
            )
        cache = cls.__dict__.get("_var_cache")
        if cache is None:
            cache = {}
            setattr(cls, "_var_cache", cache)

        key = (name, lo, hi)
        inst = cache.get(key)
        if inst is None:
            inst = super().__call__(name=name, lo=lo, hi=hi, **attrs)
            cache[key] = inst
        return inst


class DimVar(Op, metaclass=_DimVarMeta):
    name = ParamDef(kind="attribute", annotation=str)
    lo = ParamDef(kind="attribute", annotation=int)
    hi = ParamDef(kind="attribute", annotation=int)

    def __deepcopy__(self, memo):
        """Itself: one ``(name, lo, hi)`` is one instance, and copying keeps that."""
        return self

    def __add__(self, other):
        return _dim_binop(DimAdd, self, other)

    def __radd__(self, other):
        return _dim_binop(DimAdd, other, self)

    def __sub__(self, other):
        return _dim_binop(DimSub, self, other)

    def __rsub__(self, other):
        return _dim_binop(DimSub, other, self)

    def __mul__(self, other):
        return _dim_binop(DimMul, self, other)

    def __rmul__(self, other):
        return _dim_binop(DimMul, other, self)

    def __floordiv__(self, other):
        return _dim_binop(DimFloorDiv, self, other)

    def __rfloordiv__(self, other):
        return _dim_binop(DimFloorDiv, other, self)

    def __mod__(self, other):
        return _dim_binop(DimMod, self, other)

    def __rmod__(self, other):
        return _dim_binop(DimMod, other, self)


def _dim_binop(op_cls, a, b):
    """Dim binop.

    Build a dim-arithmetic Call, or ``NotImplemented`` for operands outside
    ``ShapeDim = int | DimVar | Expr``. A bool raises instead of falling back to
    ``NotImplemented``, which would let Python retry the operation as plain
    integer arithmetic and silently give the dimension the value 0 or 1.
    """
    for operand in (a, b):
        if isinstance(operand, bool):
            raise TypeError(
                f"dim arithmetic: bool operand {operand!r} is not a dimension; bool is an "
                "int subclass, so it is refused here rather than silently becoming 0 or 1"
            )
    if not all(isinstance(v, (int, DimVar, Expr)) for v in (a, b)):
        return NotImplemented
    return simplify_dim(op_cls, (a, b))


class DimAdd(Op):
    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimSub(Op):
    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimMul(Op):
    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimFloorDiv(Op):
    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimMod(Op):
    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimMin(Op):
    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


class DimMax(Op):
    a = ParamDef(kind="input")
    b = ParamDef(kind="input")


def simplify_dim(op_cls: type[Op], args: tuple) -> Expr:
    """Build dimension arithmetic without applying algebraic rules.

    Raw integers become i64 constants and bool remains invalid. The dimension
    is normalized only when it enters IR.

    See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
    """
    from .tensor_type import TensorType  # noqa: PLC0415

    ti64 = TensorType.umat_scalar()

    def _wrap(v):
        if isinstance(v, bool):
            raise TypeError(
                f"simplify_dim: bool operand {v!r} is not a ShapeDim (int / DimVar / Expr); "
                "bool is an int subclass, so it is refused here rather than silently "
                "becoming 0 or 1"
            )
        if isinstance(v, int):
            return Constant(type=ti64, value=v)
        return v

    canon_args = tuple(_wrap(a) for a in args)

    return Call(type=ti64, target=op_cls(), args=canon_args)


_DIM_OP_TYPES = (DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax)

_DIM_EXPR_VISITOR_TYPE = None
_DIM_EXPR_VISITOR = None


def _dim_expr_visitor_type():
    global _DIM_EXPR_VISITOR_TYPE
    if _DIM_EXPR_VISITOR_TYPE is None:
        from ..visitor import ExprVisitor  # noqa: PLC0415

        class _DimExprVisitor(ExprVisitor[bool]):
            def visit_DimVar(self, value: DimVar) -> bool:
                return True

            def visit_Constant(self, value: Constant) -> bool:
                return isinstance(value.value, int) and not isinstance(value.value, bool)

            def visit_Call(self, value: Call) -> bool:
                return isinstance(value.target, _DIM_OP_TYPES) and all(
                    self.visit(arg) for arg in value.args
                )

            def default_visit(self, value) -> bool:
                return False

        _DIM_EXPR_VISITOR_TYPE = _DimExprVisitor
    return _DIM_EXPR_VISITOR_TYPE


def _dim_expr_visitor():
    global _DIM_EXPR_VISITOR
    if _DIM_EXPR_VISITOR is None:
        _DIM_EXPR_VISITOR = _dim_expr_visitor_type()()
    _DIM_EXPR_VISITOR.clear()
    return _DIM_EXPR_VISITOR


def is_dim_expr(value) -> bool:
    """True iff *value* is a valid static-or-symbolic dim expression.

    True iff *value* is a valid static-or-symbolic dim expression:
    a non-bool ``int``, a ``DimVar``, an ``int``-valued ``Constant``, or a
    ``Call`` over the dim-arithmetic ops whose args all satisfy this.

    This module owns the dim-op set, so a new dim op is added beside its
    own membership here.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, (DimVar, Constant, Call)):
        return _dim_expr_visitor().visit(value)
    return False


def dim_expr(value) -> Expr:
    """Convert a dimension value to an expression without simplifying it."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        raise TypeError("dim_expr: bool is not a dimension")
    if isinstance(value, int):
        from .tensor_type import TensorType  # noqa: PLC0415

        return Constant(type=TensorType.umat_scalar(), value=value)
    if isinstance(value, DimVar):
        return simplify_dim(DimAdd, (value, 0))
    raise TypeError(f"dim_expr: expected int, DimVar, or Expr, got {type(value).__name__}")


def is_dim_op_call(value) -> bool:
    """True iff *value* is a ``Call`` over one of this module's dim ops.

    Unlike :func:`is_dim_expr` this asks only what the call computes, not what
    it computes over, so it also holds for a coordinate built on a scalar ``Var``
    that only the surrounding walk can resolve. Dim arithmetic is an address
    rather than a value some kernel produces, so a walk over compute ops uses
    this to leave it alone.
    """
    return isinstance(value, Call) and isinstance(value.target, _DIM_OP_TYPES)


def dim_min(a, b) -> Expr:
    """Build a symbolic ``min(a, b)`` dimension expression."""
    result = _dim_binop(DimMin, a, b)
    if result is NotImplemented:
        raise TypeError(
            f"dim_min: operands must be int, DimVar, or Expr, got "
            f"{type(a).__name__} and {type(b).__name__}"
        )
    return result


def dim_max(a, b) -> Expr:
    """Symbolic ``max(a, b)`` dim expression; see ``dim_min``."""
    result = _dim_binop(DimMax, a, b)
    if result is NotImplemented:
        raise TypeError(
            f"dim_max: operands must be int, DimVar, or Expr, got "
            f"{type(a).__name__} and {type(b).__name__}"
        )
    return result


def ceildiv(a, b) -> Expr:
    """Ceiling division ``(a + b - 1) // b`` as a dim expression.

    Composes existing dim-arithmetic ops — there is no dedicated ceil-div
    op. Operands may be ``int`` (non-bool), ``DimVar`` or ``Expr``; the
    result is the same ``ShapeDim`` form produced by ``simplify_dim``.
    """
    num = simplify_dim(DimSub, (simplify_dim(DimAdd, (a, b)), 1))
    return simplify_dim(DimFloorDiv, (num, b))


__all__ = [
    "DimConst",
    "DimVar",
    "DimAdd",
    "DimSub",
    "DimMul",
    "DimFloorDiv",
    "DimMod",
    "DimMin",
    "DimMax",
    "simplify_dim",
    "is_dim_expr",
    "dim_expr",
    "is_dim_op_call",
    "dim_min",
    "dim_max",
    "ceildiv",
]
