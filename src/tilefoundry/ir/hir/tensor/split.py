"""HIR ``Split`` Op: cut a tensor into several outputs along an axis."""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.types import int_tuple


@expr.register_op(dialect='tf', category='tensor')
class Split(expr.Op):
    """Multi-output op. `Call.type` is `TupleType` ([types §5](docs/spec/types.md#5-tupletype))."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    axis = expr.ParamDef(kind="attribute", annotation=int)
    num_splits = expr.ParamDef(kind="attribute", annotation=int)


def reject_redistribution(ctx, call: "Call", x_ty, axis: int, parts: int) -> None:
    """Refuse dividing an axis the layout already hands out in slices.

    Each participant would hold one whole part, and saying which one means
    knowing where that participant starts. This wave carries extents and not
    offsets, so the redistribution is refused at the contract the author wrote
    against rather than surfacing later as an analysis that cannot answer.
    """
    layout = types.shard_layout_of(x_ty.layout)
    if layout is None:
        return
    targets = types.split_target_axes(layout, x_ty.shape)
    mesh = types.flatten(layout.mesh.layout).shape if layout.mesh is not None else ()
    for mesh_axis, attr in enumerate(layout.attrs):
        divides = mesh_axis < len(mesh) and mesh[mesh_axis] > 1
        if isinstance(attr, types.Split) and divides and targets[mesh_axis] == axis:
            ctx.error(
                call,
                f"Split: axis {axis} is divided into {parts} parts and is "
                "already Split across participants, so which part a participant "
                "holds depends on an offset this analysis does not carry; "
                "reshard to a replicated layout before the split",
            )


@typeinfer.register(Split)
def _(call: "Call", ctx: "TypeInferContext") -> types.TupleType:
    x_ty = ctx.type_of(call.args[0])
    raw_axis = call.target.axis
    rank = len(x_ty.shape)
    axis = raw_axis + rank if raw_axis < 0 else raw_axis
    if not (0 <= axis < rank):
        ctx.error(call, f"axis {raw_axis} out of range for rank {rank}")
    n = call.target.num_splits
    if n <= 0:
        ctx.error(call, f"num_splits must be positive, got {n}")
    reject_redistribution(ctx, call, x_ty, axis, n)
    orig = x_ty.shape[axis]
    v = expr.static_dim_value(orig)
    if v is not None:
        if v % n != 0:
            ctx.error(call, f"axis {axis} extent {v} not divisible by {n}")
        part_len = v // n
    else:
        part_len = orig
    part_shape = list(x_ty.shape)
    part_shape[axis] = part_len
    part_shape = tuple(part_shape)
    if isinstance(x_ty.layout, types.ShardLayout):
        layout_to_tensor = int_tuple.mode_axes(int_tuple.compatible_profile(x_ty.layout.layout.shape, x_ty.shape))
        attrs = tuple(
            types.Split(layout_to_tensor[attr.axis])
            if isinstance(attr, types.Split)
            else types.Partial(attr.reduction)
            if isinstance(attr, types.Partial)
            else types.Broadcast()
            for attr in x_ty.layout.attrs
        )
        part_layout = types.ShardLayout.from_logical(
            part_shape,
            x_ty.layout.mesh,
            attrs,
            storage=x_ty.storage,
            owners=ctx.target.storage_owners
            if ctx is not None and ctx.target is not None
            else expr.StorageOwners((), {}),
        )
    else:
        part_layout = types.Layout(part_shape)
    part_ty = types.TensorType(
        shape=part_shape, dtype=x_ty.dtype, layout=part_layout, storage=x_ty.storage
    )
    return types.TupleType(fields=tuple(part_ty for _ in range(n)))


@evaluator.register(Split)
def eval_split(ctx):
    fields = ctx.result_type.fields

    def split(source):
        axis = ctx.op.axis + source.ndim if ctx.op.axis < 0 else ctx.op.axis
        n = ctx.op.num_splits
        extent = source.shape[axis]
        if extent % n:
            raise ValueError(f"Split: runtime axis {axis} extent {extent} is not divisible by {n}")
        return utils.torch.chunk(source, n, dim=axis)

    x = ctx.args[0].tensor
    if x.is_global:
        held = split(x.tensor)
    else:
        each = [(p, split(t)) for p, t in x.tensor]
        held = tuple(tuple((p, out[i]) for p, out in each) for i in range(len(fields)))
    return evaluator.TupleValue(
        tuple(evaluator.TensorValue(core.ShardTensor(field, data)) for field, data in zip(fields, held, strict=True))
    )


@access.register_relation(Split)
def split_access(call: "Call", ctx) -> tuple[access.Relation, ...]:
    """One space, the source's, with each part written on its own run of it.

    Every element of the source becomes an element of exactly one part, so the
    space to walk is the source's and each output is partial in it: the run of
    the split axis that part covers, shifted to that part's own coordinates.
    Walking a part's own coordinates instead would say every part reads the
    first one.
    """
    source = ctx.type_of(call.args[0])
    rank = len(source.shape)
    axis = call.target.axis + rank if call.target.axis < 0 else call.target.axis
    parts = call.target.num_splits
    extent = source.shape[axis]
    if not isinstance(extent, int) or isinstance(extent, bool) or extent % parts:
        raise NotImplementedError(
            f"Split access_relation: axis {axis} extent {extent!r} does not "
            f"divide into {parts} static parts"
        )
    chunk = extent // parts
    domain = ", ".join(f"d{index}" for index in range(rank))
    written = []
    for part in range(parts):
        writes = [f"d{index}" for index in range(rank)]
        begin = part * chunk
        if begin:
            writes[axis] = f"d{axis} - {begin}"
        written.append(
            access.Relation(
                isl.map(
                    f"{{ [{domain}] -> [{', '.join(writes)}] : "
                    f"{begin} <= d{axis} < {begin + chunk} }}"
                )
            )
        )

    return access.iterating(source.shape, (access.identity(rank), *written))


cost.register(Split)(cost.structure)


__all__ = ["Split"]
