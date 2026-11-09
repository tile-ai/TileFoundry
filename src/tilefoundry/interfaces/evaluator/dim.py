"""Evaluate-time resolution of a ``Dim`` (``ShapeDim``) to a concrete ``int``.

This is an evaluation-time utility (it substitutes runtime ``DimVar`` sizes and
folds the dim expression), distinct from the IR-construction / folding in
``tilefoundry.ir.dim``.

A dim expression also reaches the interpreter as an operand rather than as a
shape -- a ``Slice`` start moved off a loop's induction variable is one -- so the
same folding is registered per dim op. There it folds the values its arguments
already evaluated to instead of substituting ``DimVar`` sizes.
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from types import MappingProxyType

from tilefoundry import core
from tilefoundry.ir import core as ir_core
from tilefoundry.ir import expr

from .registry import register
from .utils import common_placements
from .value import EvalError, TensorValue

FOLDERS: MappingProxyType[type, Callable[[int, int], int]] = MappingProxyType({
    expr.DimAdd: lambda a, b: a + b,
    expr.DimSub: lambda a, b: a - b,
    expr.DimMul: lambda a, b: a * b,
    expr.DimFloorDiv: lambda a, b: a // b,
    expr.DimMod: lambda a, b: a % b,
    expr.DimMin: min,
    expr.DimMax: max,
})


class DimResolver(ir_core.ExprVisitor[int]):
    def __init__(self, bindings: dict[str, int]) -> None:
        super().__init__()
        self.bindings = bindings

    def visit_Constant(self, dim: expr.Constant, ctx=None) -> int:
        value = dim.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"resolve_dim: non-integer Constant value {value!r}")
        return value

    def visit_DimVar(self, dim: expr.DimVar, ctx=None) -> int:
        try:
            return self.bindings[dim.name]
        except KeyError:
            raise ValueError(f"resolve_dim: unbound DimVar {dim.name!r}") from None

    def visit_Call(self, dim: expr.Call, ctx=None) -> int:
        op_cls = type(dim.target)
        fold = FOLDERS.get(op_cls)
        if fold is None:
            raise ValueError(f"resolve_dim: non-dim Call target {op_cls.__name__}")
        a = self.visit(dim.args[0], ctx)
        b = self.visit(dim.args[1], ctx)
        if op_cls in (expr.DimFloorDiv, expr.DimMod) and b == 0:
            raise ValueError("resolve_dim: division/modulo by zero")
        return int(fold(a, b))

    @typing.override
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
    if isinstance(dim, (expr.Constant, expr.DimVar, expr.Call)):
        return DimResolver(bindings).visit(dim)
    raise ValueError(f"resolve_dim: unrecognised Dim {type(dim).__name__}")


def dim_op_eval(op_cls: type, fold: Callable[[int, int], int]):
    """The evaluator for one dim op in operand position.

    An operand-position dim expression is a compile-time coordinate, so its
    leaves are ``Constant`` / induction ``Var`` rather than ``DimVar`` sizes:
    the operands arrive already evaluated and only the arithmetic is left.
    """

    def evaluate(ctx):
        name = op_cls.__name__
        if len(ctx.args) != 2:
            raise EvalError(f"evaluator: {name} takes 2 operands, got {len(ctx.args)}")
        if not all(isinstance(arg, TensorValue) for arg in ctx.args):
            raise EvalError(f"evaluator: {name} operands are single integers")

        def apply(*held):
            if any(data.numel() != 1 for data in held):
                raise EvalError(f"evaluator: {name} operands are single integers")
            operands = [int(data.reshape(-1)[0].item()) for data in held]
            if op_cls in (expr.DimFloorDiv, expr.DimMod) and operands[1] == 0:
                raise EvalError(f"evaluator: {name} by zero")
            return held[0].new_tensor(int(fold(*operands)))

        a, b = ctx.args[0].tensor, ctx.args[1].tensor
        placements = common_placements(a, b)
        held = (
            apply(a.tensor, b.tensor)
            if not placements
            else tuple((p, apply(a.to_local(p), b.to_local(p))) for p in placements)
        )
        return TensorValue(core.ShardTensor(ctx.result_type, held))

    return evaluate


for op_cls, fold in FOLDERS.items():
    register(op_cls)(dim_op_eval(op_cls, fold))


__all__ = ["resolve_dim"]
