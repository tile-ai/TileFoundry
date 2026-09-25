from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import ComposedLayout, Layout, TensorType
from tilefoundry.ir.types.shard_layout import shard_layout_of
from tilefoundry.ir.types.stride import try_compact_major
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AffineAccess,
    identity_access,
    register_access_relation,
    relations_of,
    view_relations,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class Transpose(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    perm = ParamDef(kind="attribute", annotation=tuple)


def _strides(type_: TensorType) -> tuple | None:
    """The per-axis strides one position addresses this Type with."""
    layout = type_.layout
    shard = shard_layout_of(layout)
    if shard is not None:
        layout = shard.layout
    if not isinstance(layout, Layout) or len(layout.shape) != len(type_.shape):
        return None
    if layout.strides is not None:
        return tuple(layout.strides)
    return try_compact_major(tuple(layout.shape))


def _transpose_view(call: "Call", ctx) -> tuple:
    """Result axis k is source axis perm[k], stated in both sides' positions.

    A permutation walks what it reads, so the source's own axes are the
    coordinates and the permutation happens on the way out. Which positions those
    axes are is the reader's question, asked of every Op the same way.
    """
    perm = tuple(call.target.perm)
    source = ctx.type_of(call.args[0])
    rank = len(source.shape)
    writes_at = [f"d{source_axis}" for source_axis in perm]
    domain = ", ".join(f"d{index}" for index in range(rank))
    return (
        identity_access(rank),
        AffineAccess(isl.map(f"{{ [{domain}] -> [{', '.join(writes_at)}] }}")),
    )


register_access_relation(Transpose)(
    view_relations(
        0,
        _transpose_view,
        over=lambda call, ctx: ctx.type_of(call.args[0]).shape,
    )
)


@register_typeinfer(Transpose)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    """The same bytes with the axes in another order, and the strides to match.

    A tensor that states no layout is the C order every layer of this IR reads
    it as, and permuting that is not the C order of the result: a (K, M) read
    with strides (M, 1) transposes to an (M, K) view whose strides are (1, M),
    not (K, 1). So an unstated layout is written out as the strides it stands
    for and those are permuted with the shape.
    """
    x_ty = ctx.type_of(call.args[0])
    perm = call.target.perm
    if len(perm) != len(x_ty.shape):
        ctx.error(call, f"perm length {len(perm)} != rank {len(x_ty.shape)}")
    new_shape = tuple(x_ty.shape[p] for p in perm)

    new_layout = x_ty.layout
    source_shard = shard_layout_of(x_ty.layout)
    if source_shard is not None:
        relation = relations_of(call, ctx)
        derived = derive_output_shard_layout((x_ty,), relation, new_shape, fresh_strides=False)
        if derived is not None:
            new_layout = derived
    else:
        source = x_ty.layout
        if source is None:
            source = Layout(shape=tuple(x_ty.shape), strides=try_compact_major(tuple(x_ty.shape)))
            if source.strides is None:
                source = None
        if isinstance(source, Layout):
            new_layout = Layout(
                shape=tuple(source.shape[p] for p in perm),
                strides=(
                    None if source.strides is None else tuple(source.strides[p] for p in perm)
                ),
            )
        elif isinstance(source, ComposedLayout) and isinstance(source.outer, Layout):
            new_layout = ComposedLayout(
                inner=source.inner,
                offset=source.offset,
                outer=Layout(
                    shape=tuple(source.outer.shape[p] for p in perm),
                    strides=(
                        None
                        if source.outer.strides is None
                        else tuple(source.outer.strides[p] for p in perm)
                    ),
                ),
            )
    return TensorType(shape=new_shape, dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage)


@register_eval(Transpose)
def _eval_transpose(ctx):
    out = torch.permute(ctx.args[0].data, tuple(ctx.op.perm)).contiguous()
    return TensorValue(data=out, type=ctx.result_type)
