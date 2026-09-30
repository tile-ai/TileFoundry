from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import Tensor
from tilefoundry.ir.types import Layout, TensorType
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    identity_access,
    iterating,
    register_access_relation,
    relations_of,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class Transpose(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    perm = ParamDef(kind="attribute", annotation=tuple)

@register_access_relation(Transpose)
def _transpose_relations(call: "Call", ctx) -> AccessRelations:
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
        AccessRelations(
            (BoundaryRelation(identity_access(rank)),),
            (
                BoundaryRelation(
                    AffineAccess(isl.map(f"{{ [{domain}] -> [{', '.join(writes_at)}] }}"))
                ),
            ),
        ),
    )


@register_typeinfer(Transpose)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    """A new compact value with its source axes in another order."""
    x_ty = ctx.type_of(call.args[0])
    perm = call.target.perm
    if len(perm) != len(x_ty.shape):
        ctx.error(call, f"perm length {len(perm)} != rank {len(x_ty.shape)}")
    new_shape = tuple(x_ty.shape[p] for p in perm)

    new_layout = derive_output_shard_layout(
        (x_ty,), relations_of(call, ctx), new_shape, fresh_strides=True
    )
    if new_layout is None:
        new_layout = Layout(new_shape, tuple(compact_row_major(new_shape)))
    return TensorType(shape=new_shape, dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage)


@register_eval(Transpose)
def _eval_transpose(ctx):
    out = torch.permute(ctx.args[0].data, tuple(ctx.op.perm)).contiguous()
    return TensorValue(data=out, type=ctx.result_type)
