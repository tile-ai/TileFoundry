"""HIR broadcast elementwise selection."""

from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import broadcast_shapes, is_one, resolve_anchor_storage
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.hir.math.binary import _merge_layout
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, try_c_order_strides
from tilefoundry.ir.types.shard.shard_layout import Broadcast, shard_layout_of
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    iterating,
    register_access_relation,
    relations_of,
    shape_from_relation,
)
from tilefoundry.visitor_registry.isl_utility import to_domain
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class Where(Op):
    """Select values from two branches under a boolean condition."""

    condition = ParamDef(kind="input")
    input = ParamDef(kind="input")
    other = ParamDef(kind="input")


def _broadcast_all(shapes: tuple[tuple, ...]) -> tuple:
    out_shape = shapes[0]
    for shape in shapes[1:]:
        out_shape = broadcast_shapes(out_shape, shape)
    return out_shape


def _maps(shapes: tuple[tuple, ...]) -> tuple[object, tuple[AffineAccess, ...], dict]:
    out_shape = _broadcast_all(shapes)
    rank = len(out_shape)
    domain, param_map = to_domain(out_shape)
    dims = [f"d{i}" for i in range(rank)]
    source = "[" + ", ".join(dims) + "]"
    maps = []
    for shape in shapes:
        pad = rank - len(shape)
        accessed = [
            "0" if is_one(shape[i]) and not is_one(out_shape[pad + i]) else dims[pad + i]
            for i in range(len(shape))
        ]
        maps.append(AffineAccess(isl.map(f"{{ {source} -> [{', '.join(accessed)}] }}")))
    maps.append(AffineAccess(isl.map(f"{{ {source} -> [{', '.join(dims)}] }}")))
    return (domain, tuple(maps), param_map)


@register_access_relation(Where)
def _where_access_relation(call: "Call", ctx) -> AccessRelations:
    input_types = tuple(ctx.type_of(arg) for arg in call.args)
    shapes = tuple(type_.shape for type_ in input_types)
    out_shape = _broadcast_all(shapes)
    _domain, maps, _params = _maps(shapes)
    return iterating(
        out_shape,
        AccessRelations(
            inputs=tuple(BoundaryRelation(item) for item in maps[:-1]),
            outputs=(BoundaryRelation(maps[-1]),),
        ),
    )


def _data_relation(relations: AccessRelations) -> AccessRelations:
    """The same Op without its condition: the two branches and the result."""
    return AccessRelations(inputs=relations.inputs[1:3], outputs=relations.outputs)


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
            relation, _broadcast_all((condition.shape, input_.shape, other.shape))
        )
        data_relation = _data_relation(relation)
        data_shard = derive_output_shard_layout((input_, other), data_relation, out_shape)
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
        layout = Layout(shape=out_shape, strides=try_c_order_strides(out_shape))
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
