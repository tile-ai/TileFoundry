from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.hir._shard_checks import (
    reject_dynamic_shards,
    require_compatible_meshes,
    require_uniform_partial_slices,
)
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard import (
    Layout,
    shard_layout_of,
    try_c_order_strides,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    build_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class Stack(Op):
    """Variadic input op. See Concat for encoding rationale."""

    is_variadic: ClassVar[bool] = True

    inputs = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)


def _axis(call: "Call", ctx: "TypeInferContext", rank: int) -> int:
    raw_axis = call.target.axis
    axis = raw_axis + rank + 1 if raw_axis < 0 else raw_axis
    if not (0 <= axis <= rank):
        ctx.error(call, f"axis {raw_axis} out of range for input rank {rank}")
    return axis


@register_type_relation(Stack)
def _stack_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    rank = len(input_types[0].shape)
    axis = _axis(call, ctx, rank)
    output_shape = list(input_types[0].shape)
    output_shape.insert(axis, len(input_types))
    dims = [f"d{i}" for i in range(rank + 1)]
    domain_text = ", ".join(dims)
    input_text = ", ".join((*dims[:axis], *dims[axis + 1 :]))
    input_maps = tuple(
        isl.map(f"{{ [{domain_text}] -> [{input_text}] : d{axis} = {index} }}")
        for index in range(len(input_types))
    )
    output_map = isl.map(f"{{ [{domain_text}] -> [{domain_text}] }}")
    return AccessRelationResult(
        domain=build_domain(tuple(output_shape)), maps=(*input_maps, output_map)
    )


@register_typeinfer(Stack)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    if not call.args:
        ctx.error(call, "Stack requires at least one input")
    types = [ctx.type_of(a) for a in call.args]
    base = types[0]
    for index, t in enumerate(types[1:], start=1):
        if t.shape != base.shape:
            ctx.error(call, f"input {index} shape must match input 0")
        if t.dtype != base.dtype:
            ctx.error(call, f"input {index} dtype must match input 0")
    axis = _axis(call, ctx, len(base.shape))
    new_shape = list(base.shape)
    new_shape.insert(axis, len(call.args))
    new_shape = tuple(new_shape)
    reject_dynamic_shards(ctx, call, types, "Stack")
    require_compatible_meshes(ctx, call, types, "Stack")
    try:
        relation = build_relation(call, tuple(types), ctx)
        layout = derive_output_shard_layout(tuple(types), relation, new_shape, fresh_strides=True)
    except ValueError as error:
        ctx.error(
            call,
            f"cannot derive input ownership: {error}; use an explicit Reshard before Stack",
        )
    if layout is not None:
        require_uniform_partial_slices(ctx, call, types, layout, "Stack")
        if getattr(layout.layout, "strides", None) is None:
            first = next(
                index
                for index, type_ in enumerate(types)
                if shard_layout_of(type_.layout) is not None
            )
            ctx.error(
                call,
                f"input {first} ownership produces an unrepresentable result "
                "layout; use an explicit Reshard before Stack",
            )
    else:
        layout = Layout(shape=new_shape, strides=try_c_order_strides(new_shape))
    storage = resolve_anchor_storage(ctx, call, *(t.storage for t in types))
    return TensorType(shape=new_shape, dtype=base.dtype, layout=layout, storage=storage)


@register_eval(Stack)
def _eval_stack(ctx):
    data = torch.stack(tuple(arg.data for arg in ctx.args), dim=ctx.op.axis)
    return TensorValue(data=data, type=ctx.result_type)
