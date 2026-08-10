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
    ``ShapeDim = int | DimVar | Expr``.
    """

    def _ok(v):

        if isinstance(v, bool):
            return False
        return isinstance(v, (int, DimVar, Expr))

    if not (_ok(a) and _ok(b)):
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


_DIM_FOLDERS: dict[type[Op], object] = {
    DimAdd: lambda a, b: a + b,
    DimSub: lambda a, b: a - b,
    DimMul: lambda a, b: a * b,
    DimFloorDiv: lambda a, b: a // b,
    DimMod: lambda a, b: a % b,
    DimMin: min,
    DimMax: max,
}


def simplify_dim(op_cls: type[Op], args: tuple) -> Expr:
    """Fold dimension arithmetic when every operand is an integer constant.

    Raw integers canonicalize to i64 constants. Unsupported operations and
    division or modulo by zero remain calls for later verification; algebraic
    identities are not folded.

    See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
    """
    from .tensor_type import TensorType  # noqa: PLC0415

    ti64 = TensorType.meta_scalar()

    def _wrap(v):
        if isinstance(v, bool):
            raise TypeError(
                f"simplify_dim: bool operand {v!r} is not a valid "
                f"ShapeDim entry (use int / DimVar / Expr)"
            )
        if isinstance(v, int):
            return Constant(type=ti64, value=v)
        return v

    canon_args = tuple(_wrap(a) for a in args)

    fold = _DIM_FOLDERS.get(op_cls)
    if (
        fold is not None
        and len(canon_args) == 2
        and all(
            isinstance(a, Constant) and isinstance(a.value, int) and not isinstance(a.value, bool)
            for a in canon_args
        )
    ):
        a_val = int(canon_args[0].value)
        b_val = int(canon_args[1].value)
        if op_cls in (DimFloorDiv, DimMod) and b_val == 0:
            pass
        else:
            return Constant(type=ti64, value=fold(a_val, b_val))
    return Call(type=ti64, target=op_cls(), args=canon_args)


_DIM_OP_TYPES = (DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax)


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
    if isinstance(value, DimVar):
        return True
    if isinstance(value, Constant):
        return isinstance(value.value, int) and not isinstance(value.value, bool)
    if isinstance(value, Call):
        return isinstance(value.target, _DIM_OP_TYPES) and all(is_dim_expr(a) for a in value.args)
    return False


def dim_min(a, b) -> Expr:
    """Symbolic ``min`` dim expression, folded to a ``Constant`` when both operands are static.

    Symbolic ``min(a, b)`` dim expression, folded to a ``Constant`` when both
    operands are static.
    """
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
    result is the same ``ShapeDim`` form produced by ``simplify_dim`` and
    folds to a ``Constant`` when both operands are static.
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
    "dim_min",
    "dim_max",
    "ceildiv",
]
