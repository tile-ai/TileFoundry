"""HIR ``IndexSelect`` Op: select whole slices along a dimension."""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types, utils


@expr.register_op(name="index_select", dialect='tf', category='tensor')
class IndexSelect(expr.Op):
    """Select whole slices along ``dim`` using a 1-D integer ``index``."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    index = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    dim = expr.ParamDef(kind="attribute", annotation=int, default=0)
    fill_value = expr.ParamDef(kind="attribute", annotation=float | None, default=None)


def norm_dim(dim: int, rank: int, ctx=None, call=None) -> int:
    normalized = dim + rank if dim < 0 else dim
    if normalized < 0 or normalized >= rank:
        msg = f"dim {dim} out of range for rank {rank}"
        if ctx is not None:
            ctx.error(call, msg)
        raise ValueError(f"IndexSelect: {msg}")
    return normalized


def index_select_shard_layout(call, ctx, x_ty, dim: int, out_shape: tuple):
    """Derive a natural contiguous shard layout for a whole-slice selection.

    Broadcast and Partial states carry through. A Split on the selected dim
    becomes ``Partial(sum)``. Composed or ambiguous multi-Split layouts fail
    closed.
    """
    sl = x_ty.layout
    if isinstance(sl, types.Layout):
        return types.Layout(out_shape)
    if not isinstance(sl, types.ShardLayout):
        return sl
    if not isinstance(sl.layout, types.Layout):
        ctx.error(
            call,
            f"dim {dim} has a composed shard layout; cannot derive an output layout",
        )
    targets = types.split_target_axes(sl, tuple(x_ty.shape))
    splits = [(i, t) for i, t in enumerate(targets) if t is not None]
    on_dim = [i for i, target in splits if target == dim]
    if on_dim and len(splits) > 1:
        ctx.error(
            call,
            f"dim {dim} index_select over a shard layout with multiple Split "
            "axes including the selected dim; cannot derive an output layout",
        )
    natural = types.Layout(shape=out_shape, strides=types.compact_col_major(out_shape))
    if on_dim:
        mesh_idx = on_dim[0]
        new_attrs = tuple(
            types.Partial(reduction="sum") if i == mesh_idx else a for i, a in enumerate(sl.attrs)
        )
        return types.ShardLayout(layout=natural, attrs=new_attrs, mesh=sl.mesh)

    new_attrs = tuple(
        types.Split(axis=target) if isinstance(attr, types.Split) else attr
        for attr, target in zip(sl.attrs, targets)
    )
    return types.ShardLayout(layout=natural, attrs=new_attrs, mesh=sl.mesh)


@typeinfer.register(IndexSelect)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    index_ty = ctx.type_of(call.args[1])
    if len(index_ty.shape) != 1:
        ctx.error(call, f"index must be 1-D, got shape {index_ty.shape}")
    if index_ty.dtype not in (expr.DType.i32, expr.DType.i64):
        ctx.error(call, f"index must have dtype i32 or i64, got {index_ty.dtype}")
    dim = norm_dim(call.target.dim, len(x_ty.shape), ctx, call)

    new_shape = list(x_ty.shape)
    new_shape[dim] = index_ty.shape[0]
    new_layout = index_select_shard_layout(call, ctx, x_ty, dim, tuple(new_shape))
    return types.TensorType(
        shape=tuple(new_shape), dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage
    )


@access.register_relation(IndexSelect)
def index_select_access_relation(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """GLOBAL level: IndexSelect pulls one source slice per index element.

    Every boundary is asked about the same coordinates, the result's own, so the
    source is read where the result says on the axes it keeps and anywhere the
    selected axis could legally name on the one it does not -- the deciding value
    lives in the index, not here. The index itself is read at the coordinate the
    result reached, one element per selected slice.
    """
    source_ty, index_ty = ctx.type_of(call.args[0]), ctx.type_of(call.args[1])
    axis = norm_dim(call.target.dim, len(source_ty.shape))
    return access.gather_relations(source_ty, index_ty, axis)


@evaluator.register(IndexSelect)
def eval_index_select(ctx):
    def index_select(x, index):
        return evaluator.gather(x, index, norm_dim(ctx.op.dim, x.dim()), ctx.op.fill_value)

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        index_select(*(x.tensor for x in xs))
        if not placements
        else tuple((p, index_select(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["IndexSelect", "norm_dim"]


@cost.register(IndexSelect)
def index_select_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    index = cost.input_types(call, ctx)[1]
    rows = utils.tensor_bytes(cost.output_type(call, ctx))
    return cost.Cost(
        {},
        (
            cost.TrafficBytes(read=rows),
            cost.TrafficBytes(read=utils.tensor_bytes(index)),
            cost.TrafficBytes(write=rows),
        ),
    )
