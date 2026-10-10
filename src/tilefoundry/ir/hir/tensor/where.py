"""HIR broadcast elementwise selection."""

from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.hir.math.binary import _merge_layout
from tilefoundry.ir.types import DType, Layout, TensorType
from tilefoundry.ir.types.shard_layout import Broadcast, shard_layout_of
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.stride import try_compact_major
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    broadcast_all,
    broadcast_relations,
    iterating,
    register_access_relation,
    relations_of,
    shape_from_relation,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class Where(Op):
    """Select values from two branches under a boolean condition."""

    condition = ParamDef(kind="input")
    input = ParamDef(kind="input")
    other = ParamDef(kind="input")


@register_access_relation(Where)
def _where_access_relation(call: "Call", ctx) -> tuple[AccessRelation, ...]:
    input_types = tuple(ctx.type_of(arg) for arg in call.args)
    shapes = tuple(type_.shape for type_ in input_types)
    return iterating(broadcast_all(shapes), broadcast_relations(shapes))


@register_typeinfer(Where)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    condition, input_, other = (ctx.type_of(arg) for arg in call.args)
    if condition.dtype != DType.bool:
        ctx.error(call, f"condition must have bool dtype, got {condition.dtype}")
    if input_.dtype != other.dtype:
        ctx.error(
            call,
            f"data branch dtype mismatch ({input_.dtype.name} vs {other.dtype.name})",
        )
    for name, type_ in (("condition", condition), ("input", input_), ("other", other)):
        reject_partials(ctx, call, name, type_.layout)

    try:
        relation = relations_of(call, ctx)
        out_shape = shape_from_relation(
            relation[len(call.args)],
            broadcast_all((condition.shape, input_.shape, other.shape)),
        )
        data_shard = derive_output_shard_layout((input_, other), relation[1:], out_shape)
        layout = (
            data_shard
            if data_shard is not None
            else _merge_layout(
                shard_layout_of(input_.layout) or input_.layout,
                shard_layout_of(other.layout) or other.layout,
                out_shape,
            )
        )

        condition_shard = shard_layout_of(condition.layout)
        if condition_shard is not None and any(
            not isinstance(attr, Broadcast) for attr in condition_shard.attrs
        ):
            combined = derive_output_shard_layout((condition, input_, other), relation, out_shape)
            if combined != layout:
                ctx.error(
                    call,
                    "condition sharding does not match the data branches; "
                    "reshard the condition to their distribution",
                )
    except ValueError as error:
        ctx.error(call, f"Where: {error}")

    storage = resolve_anchor_storage(ctx, call, input_.storage, other.storage)
    if layout is None and storage in (StorageKind.RMEM, StorageKind.SMEM) and out_shape:
        layout = Layout(shape=out_shape, strides=try_compact_major(out_shape))
    return TensorType(
        shape=out_shape,
        dtype=input_.dtype,
        layout=layout,
        storage=storage,
    )


@register_eval(Where)
def _eval_where(ctx):
    data = torch.where(
        ctx.args[0].data,
        ctx.args[1].data,
        ctx.args[2].data,
    )
    return TensorValue(data=data, type=ctx.result_type)


__all__ = ["Where"]
