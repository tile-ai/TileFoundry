"""Evaluate-time resolution of a ``Dim`` (``ShapeDim``) to a concrete ``int``.

This is an evaluation-time utility (it substitutes runtime ``DimVar`` sizes and
folds the dim expression), distinct from the IR-construction / folding in
``tilefoundry.ir.types.dim``.

A dim expression also reaches the interpreter as an operand rather than as a
shape -- a ``Slice`` start moved off a loop's induction variable is one -- so the
same folding is registered per dim op. There it folds the values its arguments
already evaluated to instead of substituting ``DimVar`` sizes.
"""
from __future__ import annotations

from typing import Callable

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue
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
from tilefoundry.ir.visitor import ExprVisitor

_FOLDERS: dict[type, Callable[[int, int], int]] = {
    DimAdd: lambda a, b: a + b,
    DimSub: lambda a, b: a - b,
    DimMul: lambda a, b: a * b,
    DimFloorDiv: lambda a, b: a // b,
    DimMod: lambda a, b: a % b,
    DimMin: min,
    DimMax: max,
}


class _DimResolver(ExprVisitor[int]):
    def __init__(self, bindings: dict[str, int]) -> None:
        super().__init__()
        self.bindings = bindings

    def visit_Constant(self, dim: Constant, ctx=None) -> int:
        value = dim.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"resolve_dim: non-integer Constant value {value!r}")
        return value

    def visit_DimVar(self, dim: DimVar, ctx=None) -> int:
        try:
            return self.bindings[dim.name]
        except KeyError:
            raise ValueError(f"resolve_dim: unbound DimVar {dim.name!r}") from None

    def visit_Call(self, dim: Call, ctx=None) -> int:
        op_cls = type(dim.target)
        fold = _FOLDERS.get(op_cls)
        if fold is None:
            raise ValueError(f"resolve_dim: non-dim Call target {op_cls.__name__}")
        a = self.visit(dim.args[0], ctx)
        b = self.visit(dim.args[1], ctx)
        if op_cls in (DimFloorDiv, DimMod) and b == 0:
            raise ValueError("resolve_dim: division/modulo by zero")
        return int(fold(a, b))

    def default_visit(self, dim, ctx=None) -> int:
        raise ValueError(f"resolve_dim: unrecognised Dim {type(dim).__name__}")


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
    if isinstance(dim, (Constant, DimVar, Call)):
        return _DimResolver(bindings).visit(dim)
    raise ValueError(f"resolve_dim: unrecognised Dim {type(dim).__name__}")


def _dim_op_eval(op_cls: type, fold: Callable[[int, int], int]):
    """The evaluator for one dim op in operand position.

    An operand-position dim expression is a compile-time coordinate, so its
    leaves are ``Constant`` / induction ``Var`` rather than ``DimVar`` sizes:
    the operands arrive already evaluated and only the arithmetic is left.
    """

    def _eval(ctx):
        name = op_cls.__name__
        if len(ctx.args) != 2:
            raise EvalError(f"evaluator: {name} takes 2 operands, got {len(ctx.args)}")
        operands = []
        for arg in ctx.args:
            if not isinstance(arg, TensorValue) or arg.data.numel() != 1:
                raise EvalError(f"evaluator: {name} operands are single integers")
            operands.append(int(arg.data.reshape(-1)[0].item()))
        if op_cls in (DimFloorDiv, DimMod) and operands[1] == 0:
            raise EvalError(f"evaluator: {name} by zero")
        return TensorValue(
            data=ctx.args[0].data.new_tensor(int(fold(*operands))),
            type=ctx.result_type,
        )

    return _eval


for _op_cls, _fold in _FOLDERS.items():
    register_eval(_op_cls)(_dim_op_eval(_op_cls, _fold))


__all__ = ["resolve_dim"]
