"""Evaluate-time resolution of a ``Dim`` (``ShapeDim``) to a concrete ``int``.

This is an evaluation-time utility (it substitutes runtime ``DimVar`` sizes and
folds the dim expression), distinct from the IR-construction / folding in
``tilefoundry.ir.types.dim``.
"""
from __future__ import annotations

from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
)

_FOLDERS = {
    DimAdd: lambda a, b: a + b,
    DimSub: lambda a, b: a - b,
    DimMul: lambda a, b: a * b,
    DimFloorDiv: lambda a, b: a // b,
    DimMod: lambda a, b: a % b,
    DimMin: min,
    DimMax: max,
}


def resolve_dim(dim, bindings: dict[str, int]) -> int:
    """Resolve a ``Dim`` to a concrete ``int`` given a ``DimVar`` name → size binding.

    Resolve a ``Dim`` (``ShapeDim``) to a concrete ``int`` given a ``DimVar``
    name → size binding.

    Raises ``ValueError`` on an unbound ``DimVar``, a ``bool`` / non-integer
    leaf, an unrecognised dim form, or division / modulo by zero.
    """
    if isinstance(dim, bool):
        raise ValueError(f"resolve_dim: bool {dim!r} is not a valid Dim")
    if isinstance(dim, int):
        return dim
    if isinstance(dim, Constant):
        v = dim.value
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"resolve_dim: non-integer Constant value {v!r}")
        return v
    if isinstance(dim, DimVar):
        try:
            return bindings[dim.name]
        except KeyError:
            raise ValueError(f"resolve_dim: unbound DimVar {dim.name!r}") from None
    if isinstance(dim, Call):
        op_cls = type(dim.target)
        fold = _FOLDERS.get(op_cls)
        if fold is None:
            raise ValueError(f"resolve_dim: non-dim Call target {op_cls.__name__}")
        a = resolve_dim(dim.args[0], bindings)
        b = resolve_dim(dim.args[1], bindings)
        if op_cls in (DimFloorDiv, DimMod) and b == 0:
            raise ValueError("resolve_dim: division/modulo by zero")
        return int(fold(a, b))
    raise ValueError(f"resolve_dim: unrecognised Dim {type(dim).__name__}")


__all__ = ["resolve_dim"]
