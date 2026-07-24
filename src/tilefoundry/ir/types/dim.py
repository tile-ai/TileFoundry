from __future__ import annotations

from ..core.expr import Call, Constant, Expr
from ..core.op import Op
from ..core.param_def import ParamDef


class DimConst(Op):
    value = ParamDef(kind="attribute", annotation=int)


class _DimVarMeta(type(Op)):
    def __call__(cls, name=None, lo=None, hi=None, **attrs):
        # Accept both positional and keyword forms.
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
        # Canonical identity is per ``(name, lo, hi)``. Same-name with
        # different bounds simply produces a distinct canonical object;
        # signature-scoped conflict detection lives in HIR
        # ``verify_function``.
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

    # Author-facing dim arithmetic sugar — lets DSL annotations write
    # ``Tensor[(..., CTX_LEN + 1, ...), "bf16"]`` and have the shape
    # entry land as a ``simplify_dim(DimAdd, ...)`` ``Call`` (i.e. an
    # ``Expr``, which is a valid ``ShapeDim``). Symmetric ``__radd__``
    # handles ``1 + CTX_LEN``.
    def __add__(self, other):
        return _dim_binop(DimAdd, self, other)

    def __radd__(self, other):
        return _dim_binop(DimAdd, other, self)

    # Floor-division counterpart to __add__ above — lets a dynamic-k
    # attribute (e.g. ``TopK.k``) be written as ``CTX_LEN // 4`` and have
    # the entry land as a ``simplify_dim(DimFloorDiv, ...)`` ``Call`` (i.e.
    # an ``Expr``, which is a valid ``ShapeDim``). Symmetric
    # ``__rfloordiv__`` handles ``4 // CTX_LEN``.
    def __floordiv__(self, other):
        return _dim_binop(DimFloorDiv, self, other)

    def __rfloordiv__(self, other):
        return _dim_binop(DimFloorDiv, other, self)


def _dim_binop(op_cls, a, b):
    """Build a dim-arithmetic Call from ``int`` (non-bool), ``DimVar``,
    or ``Expr`` operands. Anything else returns ``NotImplemented`` so
    Python falls through to the normal ``TypeError`` for unsupported
    operand types, preserving the ``ShapeDim = int | DimVar | Expr``
    contract and preventing malformed IR. Operand canonicalisation
    (int → ``Constant``) happens once, inside ``simplify_dim``.
    """
    def _ok(v):
        # ``bool`` is a subclass of ``int`` — reject explicitly so
        # ``CTX_LEN + True`` does not silently become ``CTX_LEN + 1``.
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
    DimFloorDiv: lambda a, b: a // b,   # b == 0 guarded below
    DimMod: lambda a, b: a % b,         # b == 0 guarded below
    DimMin: min,
    DimMax: max,
}


def simplify_dim(op_cls: type[Op], args: tuple) -> Expr:
    """Construction-time folding for dim arithmetic ``Call``s.

    Returns a folded ``Constant`` when *op_cls* admits folding and
    every entry of *args* is an ``int``-valued ``Constant`` (or a
    raw ``int`` literal — both forms are accepted; raw ``int``
    entries are canonicalised to ``Constant(i64, value)`` so the
    produced ``Call`` always carries ``Expr`` args).
    Otherwise returns ``Call(target=op_cls(), args=<canonicalised>,
    type=<i64 scalar>)``.

    Division / modulo by zero is **not** folded — the original
    ``Call`` is preserved so a later verify pass can flag the
    error. No algebraic identity folding (``x + 0`` etc.).
    """
    from .tensor_type import TensorType  # avoid import cycle  # noqa: PLC0415

    ti64 = TensorType.meta_scalar()

    # Canonicalise raw ``int`` entries (common in ``TensorType.shape``
    # static dims) to ``Constant(i64, value)`` so the produced ``Call``
    # always carries ``Expr`` args. This keeps downstream consumers
    # (typeinfer, verifier, formatter, structural type equality)
    # walking a single canonical form regardless of whether the dim
    # expression was authored via ``DimVar.__add__`` (already wrapped)
    # or derived from a shape with raw-``int`` entries (e.g. Concat
    # axis-2 with a ``1`` static dim).
    #
    # ``bool`` is a subclass of ``int`` — explicitly reject it so a
    # stray ``True``/``False`` in a ShapeDim entry can never produce
    # malformed IR like ``Call(DimAdd, args=(True, DimVar))``. The
    # DSL surface (``DimVar.__add__``) also rejects bool; this
    # closes the same gap on the IR-construction side.
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
            isinstance(a, Constant)
            and isinstance(a.value, int)
            and not isinstance(a.value, bool)
            for a in canon_args
        )
    ):
        a_val = int(canon_args[0].value)
        b_val = int(canon_args[1].value)
        if op_cls in (DimFloorDiv, DimMod) and b_val == 0:
            # Preserve original Call so verify can flag div/mod by zero.
            pass
        else:
            return Constant(type=ti64, value=fold(a_val, b_val))
    return Call(type=ti64, target=op_cls(), args=canon_args)


_DIM_OP_TYPES = (DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax)


def is_dim_expr(value) -> bool:
    """True iff *value* is a valid static-or-symbolic dim expression:
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
        return isinstance(value.target, _DIM_OP_TYPES) and all(
            is_dim_expr(a) for a in value.args
        )
    return False


def dim_min(a, b) -> Expr:
    """Symbolic ``min(a, b)`` dim expression — the ``min``/``max`` counterpart
    to ``DimVar.__add__`` for forms with no natural infix operator. Same
    ``int``/``DimVar``/``Expr`` operands as ``_dim_binop``, folding to a
    ``Constant`` when both are static. Raises ``TypeError`` on any other
    operand type (a plain function has no ``NotImplemented`` fallback like an
    overloaded operator does).
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
