from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import Layout, TensorType
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    iterating,
    register_access_relation,
    relations_of,
)
from tilefoundry.visitor_registry.buffer_alias import register_buffer_alias
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout

from ._view_layout import derive_view_layout


@register_op
class Transpose(Op):
    """Permute a tensor's axes.

    A new compact value by default. With ``view``, the result re-addresses the
    source's own bytes through its permuted strides, so a tile staged once can be
    read in either orientation; only an addressable (``smem`` or ``gmem``) source
    has bytes to re-address.
    """

    x = ParamDef(kind="input", pattern=is_ranked_tensor())
    perm = ParamDef(kind="attribute", annotation=tuple)
    view = ParamDef(kind="attribute", annotation=bool, default=False)

@register_access_relation(Transpose)
def _transpose_relations(call: "Call", ctx) -> tuple[AccessRelation, ...]:
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
    return iterating(
        source.shape,
        (
            identity_access(rank),
            AccessRelation(isl.map(f"{{ [{domain}] -> [{', '.join(writes_at)}] }}")),
        ),
    )


@register_typeinfer(Transpose)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    """Permute the source's axes: a new compact value, or with ``view`` the source."""
    x_ty = ctx.type_of(call.args[0])
    perm = call.target.perm
    if len(perm) != len(x_ty.shape):
        ctx.error(call, f"perm length {len(perm)} != rank {len(x_ty.shape)}")
    new_shape = tuple(x_ty.shape[p] for p in perm)
    if call.target.view:
        return _transposed_view(call, ctx, x_ty, tuple(perm), new_shape)

    new_layout = derive_output_shard_layout(
        (x_ty,), relations_of(call, ctx), new_shape, fresh_strides=True
    )
    if new_layout is None:
        new_layout = Layout(new_shape, tuple(compact_row_major(new_shape)))
    return TensorType(shape=new_shape, dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage)


def _transposed_view(call, ctx, x_ty: TensorType, perm: tuple, new_shape: tuple) -> TensorType:
    """The source's bytes in another axis order: its own strides, permuted."""
    if x_ty.storage not in (StorageKind.SMEM, StorageKind.GMEM):
        ctx.error(
            call,
            f"a transposed view re-addresses bytes, and a {x_ty.storage} value has none "
            "to re-address; transpose it into a new value instead",
        )
    if isinstance(x_ty.layout, ShardLayout):
        ctx.error(call, "a transposed view of a distributed value is not supported")

    def permuted(layout: Layout) -> Layout | None:
        if layout.strides is None:
            return None
        return Layout(
            tuple(layout.shape[p] for p in perm), tuple(layout.strides[p] for p in perm)
        )

    layout = derive_view_layout(x_ty, new_shape, permuted)
    if layout is None or getattr(getattr(layout, "outer", layout), "strides", None) is None:
        ctx.error(call, f"cannot state the permuted strides of {x_ty.layout}")
    return TensorType(shape=new_shape, dtype=x_ty.dtype, layout=layout, storage=x_ty.storage)


register_buffer_alias(Transpose, Transpose.x, when=lambda call: bool(call.target.view))


@register_eval(Transpose)
def _eval_transpose(ctx):
    out = torch.permute(ctx.args[0].data, tuple(ctx.op.perm)).contiguous()
    return TensorValue(data=out, type=ctx.result_type)
