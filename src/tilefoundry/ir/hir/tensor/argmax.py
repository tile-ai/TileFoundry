"""ArgMax HIR primitive.

SGLang baseline kernel H3 (greedy sampling). Returns int64 indices along the
reduction axis; ``keepdim=False``.

"""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect='tf', category='tensor')
class ArgMax(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    axis = expr.ParamDef(kind="attribute", annotation=int, default=-1)


@typeinfer.register(ArgMax)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        ctx.error(call, "x must be at least rank-1")
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        ctx.error(call, f"axis {call.target.axis} out of range for rank {rank}")

    source_shard = types.shard_layout_of(x_ty.layout)
    if source_shard is not None:
        targets = types.split_target_axes(source_shard, x_ty.shape)
        if any(
            isinstance(attr, types.Split) and targets[mesh_axis] == axis
            for mesh_axis, attr in enumerate(source_shard.attrs)
        ):
            ctx.error(
                call,
                f"reduction axis {axis} must not be Split-sharded; use an "
                "explicit Reshard before ArgMax",
            )
    ir_utils.reject_partials("x", x_ty.layout)
    out_shape = tuple(d for i, d in enumerate(x_ty.shape) if i != axis)
    new_layout = types.Layout(out_shape)
    if source_shard is not None:
        relation = access.relations_of(call, ctx)
        derived = typeinfer.derive_output_shard_layout(
            (x_ty,),
            relation,
            out_shape,
            complete_reduction_dims=frozenset({axis}),
            fresh_strides=True,
            ctx=ctx,
            storage=x_ty.storage,
        )
        new_layout = (
            derived
            if derived is not None
            else types.ShardLayout.from_logical(
                out_shape,
                source_shard.mesh,
                source_shard.attrs,
                storage=x_ty.storage,
                owners=ctx.target.storage_owners
                if ctx is not None and ctx.target is not None
                else expr.StorageOwners((), {}),
            )
        )
    return types.TensorType(
        shape=out_shape,
        dtype=expr.DType.i64,
        layout=new_layout,
        storage=x_ty.storage,
    )


@evaluator.register(ArgMax)
def eval_argmax(ctx):
    def argmax(x):
        return utils.torch.argmax(x, dim=ctx.op.axis)

    x = ctx.args[0].tensor
    held = argmax(x.tensor) if x.is_global else tuple((p, argmax(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@access.register_relation(ArgMax)
def argmax_access_relation(call: "expr.Call", ctx: "TypeInferContext") -> tuple[access.Relation, ...]:
    """A scan walks what it reads, and collapses the axis it scanned on the way out.

    The coordinates are the source's own, because reading every element of the
    reduced axis is the whole of what this does. The result names one fewer of
    them, which is the collapse.
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    dims = [f"d{index}" for index in range(rank)]
    walked = ", ".join(dims)
    kept = ", ".join(dim for index, dim in enumerate(dims) if index != axis)
    return access.iterating(
        x_ty.shape,
        (access.identity(rank), access.Relation(isl.map(f"{{ [{walked}] -> [{kept}] }}"))),
    )


__all__ = ["ArgMax"]


@cost.register(ArgMax)
def argmax_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    inputs = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    return cost.Cost({}, cost.traffic(inputs, output))
