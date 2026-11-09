"""KV-cache update: write ``new[:, :s]`` at ``cache[:, cur_pos : cur_pos + s]``.

Contract and constraints: `spec hir § CacheUpdate`.
"""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types, utils


@expr.register_op(name="cache_update", dialect='tf', category='tensor')
class CacheUpdate(expr.Op):
    """Cache-update HIR operation; see `spec hir § CacheUpdate`."""

    cache = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    cur_pos = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    s = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    new = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


def rows_limit(cache: tuple, supplied: tuple) -> int | None:
    """The most rows one call may write: the fewer of what each side states."""
    stated = [
        value
        for value in (
            static(cache[1] if len(cache) > 1 else None),
            static(supplied[1] if len(supplied) > 1 else None),
        )
        if value is not None
    ]
    return min(stated) if stated else None


def static(extent) -> int | None:
    """One extent as a number, when it is one."""
    return extent if isinstance(extent, int) and not isinstance(extent, bool) else None


def row_count(node) -> object:
    """How many rows this update writes: the number, or the value that says it.

    ``s`` is a runtime scalar, so when it is not written down the answer is the
    operand itself -- a relation can carry it as a parameter and a reader can
    bind it, which is what keeps this a quantity instead of charging the whole
    cache for a handful of rows.
    """
    if isinstance(node, expr.Constant) and isinstance(node.value, int):
        return int(node.value)
    return node


def row_limit(offsets: tuple, extents: tuple, limit: int | None) -> tuple:
    """The most the row window may extend, at the position holding the rows.

    A window is stated per logical axis and projected onto positions, so the
    ceiling has to land on the one position that varies over the rows -- the same
    one the extent did.
    """
    if limit is None:
        return ()
    placed = [None] * len(extents)
    for position, extent in enumerate(extents):
        if extent is not offsets[position] and not isinstance(extent, int):
            placed[position] = limit
    return tuple(placed)


@access.register_relation(CacheUpdate)
def cache_update_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """The result is the cache with ``s`` rows replaced at ``cur_pos``.

    How many rows move is ``s`` and where they land is ``cur_pos``; only the
    first is a quantity, and it is the same one wherever the second points. The
    two controls are rank-0, so their own boundaries reach one number each.
    Everything is said in the cache's own axes: which positions those are is the
    reader's question, and answering it here would be answering it twice.
    """
    cache = tuple(ctx.type_of(call.args[0]).shape)
    logical_new = ctx.type_of(call.args[3])
    supplied = tuple(logical_new.shape)
    rank = len(cache)
    rows = row_count(call.args[2])
    start = (
        int(call.args[1].value)
        if isinstance(call.args[1], expr.Constant) and isinstance(call.args[1].value, int)
        else call.args[1]
    )
    offsets = (0, start, *(0 for _ in cache[2:]))
    extents = (cache[0], rows, *cache[2:])
    limit = rows_limit(cache, supplied)
    ceilings = (None, limit, *(None for _ in cache[2:]))
    complement, reached = access.placed_window(offsets, extents, rank, ceilings, cache)
    return access.iterating(
        cache,
        (
            complement,
            access.control_read(rank, ctx, call.args[1]),
            access.control_read(rank, ctx, call.args[2]),
            access.window_source(
                offsets,
                rank,
                logical_new,
                logical_new,
                {axis: f"d{axis}" for axis in range(rank)},
                (None, rows),
                ceilings,
            ),
            reached,
        ),
    )


def is_scalar(shape) -> bool:
    """A scalar tensor: rank 0, or every dim is the literal 1."""
    return all(
        (isinstance(d, int) and d == 1) or (isinstance(d, expr.Constant) and d.value == 1) for d in shape
    )


@typeinfer.register(CacheUpdate)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    cache_ty = ctx.type_of(call.args[0])
    cur_ty = ctx.type_of(call.args[1])
    s_ty = ctx.type_of(call.args[2])
    new_ty = ctx.type_of(call.args[3])
    if len(cache_ty.shape) != 4 or len(new_ty.shape) != 4:
        ctx.error(call, "cache and new must be rank-4 [B, len, kv_heads, head_dim]")
    if cache_ty.dtype != new_ty.dtype:
        ctx.error(call, f"cache/new dtype mismatch {cache_ty.dtype} vs {new_ty.dtype}")
    utils.require_matching_partial_state(cache_ty, new_ty, "cache", "new")
    for ax, label in ((0, "B"), (2, "kv_heads"), (3, "head_dim")):
        if cache_ty.shape[ax] != new_ty.shape[ax]:
            ctx.error(
                call,
                f"cache/new {label} mismatch: {cache_ty.shape[ax]} vs {new_ty.shape[ax]}",
            )
    for t, name in ((cur_ty, "cur_pos"), (s_ty, "s")):
        if t.dtype != expr.DType.i32:
            ctx.error(call, f"{name} must be an i32 scalar, got dtype {t.dtype}")
        if not is_scalar(t.shape):
            ctx.error(call, f"{name} must be a scalar, got shape {t.shape}")
    cap, s_cap = cache_ty.shape[1], new_ty.shape[1]
    if isinstance(cap, int) and isinstance(s_cap, int) and s_cap > cap:
        ctx.error(call, f"S_CAP {s_cap} exceeds cache capacity {cap}")
    reject_split_rows(ctx, call, cache_ty, new_ty)
    return cache_ty


def divided_axes(type_) -> set[int]:
    """Which logical axes a layout really hands out in slices.

    A `Split` over a mesh axis of one gives the whole axis to the only
    participant there is, so it divides nothing.
    """
    layout = types.shard_layout_of(type_.layout)
    if layout is None:
        return set()
    targets = types.split_target_axes(layout, type_.shape)
    mesh = types.flatten(layout.mesh.layout).shape if layout.mesh is not None else ()
    return {
        targets[mesh_axis]
        for mesh_axis, attr in enumerate(layout.attrs)
        if isinstance(attr, types.Split)
        and mesh_axis < len(mesh)
        and mesh[mesh_axis] > 1
        and targets[mesh_axis] is not None
    }


def reject_split_rows(ctx, call: "expr.Call", cache_ty, new_ty) -> None:
    """Refuse an update whose two sides do not own the same thing.

    `cur_pos` is stated against the whole row axis, so a participant holding a
    slice of it would need its own offset to say which of its rows the update
    covers. Both sides are asked, because splitting the rows of `new` while the
    cache stays whole is the same question from the other end.

    Every other axis may be split, and the corpus does split the batch -- but
    only when both sides agree on it. One side sharded and the other not means
    a participant writes rows it does not hold, or holds rows nobody wrote.
    """
    held, supplied = divided_axes(cache_ty), divided_axes(new_ty)
    if 1 in held or 1 in supplied:
        ctx.error(
            call,
            "CacheUpdate: the row axis is Split across participants, and "
            "cur_pos is stated against the whole of it, so which rows a "
            "participant writes depends on an offset the projection does "
            "not carry; reshard before the update",
        )
    disagreed = held ^ supplied
    if disagreed:
        axis = min(disagreed)
        ctx.error(
            call,
            f"CacheUpdate: axis {axis} is Split on one side and not the other, "
            "so a participant would write rows it does not hold; give cache and "
            "new the same layout before the update",
        )


def updated_cache(cache, cur_pos, s, new):
    cur_pos = int(cur_pos.reshape(-1)[0].item())
    s = int(s.reshape(-1)[0].item())
    capacity, s_cap = cache.shape[1], new.shape[1]
    if cur_pos < 0:
        raise evaluator.EvalError(f"cache_update: cur_pos {cur_pos} must be >= 0")
    if not (1 <= s <= s_cap):
        raise evaluator.EvalError(f"cache_update: s {s} must satisfy 1 <= s <= {s_cap}")
    if cur_pos + s > capacity:
        raise evaluator.EvalError(
            f"cache_update: cur_pos + s ({cur_pos + s}) exceeds cache capacity {capacity}"
        )
    out = cache.clone()
    out[:, cur_pos : cur_pos + s] = new[:, :s].to(out.dtype)
    return out


@evaluator.register(CacheUpdate)
def eval_cache_update(ctx):
    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        updated_cache(*(x.tensor for x in xs))
        if not placements
        else tuple((p, updated_cache(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["CacheUpdate"]


def by_logical_axis(local, logical) -> list:
    """One extent per logical axis of a value, folding the positions it factored."""
    extents = [1] * len(logical.shape)
    for position, owner in enumerate(access.logical_axes_of(local, logical)):
        extents[owner] *= local.shape[position]
    return extents


@cost.register(CacheUpdate)
def cache_update_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """Charge the statically bounded new window, never the whole cache."""
    _, cur_pos, s, new = cost.input_types(call, ctx)
    window = utils.tensor_bytes(new)
    return cost.Cost(
        {},
        (
            cost.TrafficBytes(),
            cost.TrafficBytes(read=utils.tensor_bytes(cur_pos)),
            cost.TrafficBytes(read=utils.tensor_bytes(s)),
            cost.TrafficBytes(read=window),
            cost.TrafficBytes(write=window),
        ),
    )
