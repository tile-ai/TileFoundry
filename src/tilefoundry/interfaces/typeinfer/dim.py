"""Type inference of the ``Dim*`` nodes: each is a scalar of the meta dtype."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir import expr, types

from .registry import register

__all__: list[str] = []


def infer_dim(call, ctx):
    """A dimension expression is a scalar number; it is Broadcast when every operand is.

    A literal operand reads as Broadcast
    ([shard §6.1](docs/spec/shard.md#61-shard-attr-semantics)).
    """
    plain = types.TensorType.from_umat_scalar()
    shards = tuple(
        types.shard_layout_of(ctx.type_of(arg).layout)
        for arg in getattr(call, "args", ())
        if isinstance(arg, expr.Expr) and not isinstance(arg, expr.Constant)
    )
    if not shards or not all(
        shard is not None and all(isinstance(attr, types.Broadcast) for attr in shard.attrs) for shard in shards
    ):
        return plain
    return replace(plain, layout=types.ShardLayout(plain.layout, shards[0].attrs, shards[0].mesh))


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
