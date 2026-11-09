"""Type inference of the ``Dim*`` nodes: each is a scalar of the meta dtype."""

from __future__ import annotations

from tilefoundry.ir import expr, types

from .registry import register

__all__: list[str] = []


def infer_dim(call, ctx):
    """A dimension expression is a scalar number, whatever its operands are."""
    return types.TensorType.from_umat_scalar()


for node_cls in (
    expr.DimConst,
    expr.DimVar,
    expr.DimAdd,
    expr.DimSub,
    expr.DimMul,
    expr.DimFloorDiv,
    expr.DimMod,
    expr.DimMin,
    expr.DimMax,
):
    register(node_cls)(infer_dim)
