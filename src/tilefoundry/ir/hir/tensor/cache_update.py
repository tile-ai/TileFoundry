"""KV-cache update: write ``new[:, :s]`` at ``cache[:, cur_pos : cur_pos + s]``.

Contract and constraints: `spec hir § CacheUpdate`.
"""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError, TensorValue
from tilefoundry.ir.core import Constant, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import require_matching_partial_state
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import shard_layout_of
from tilefoundry.ir.types.shard.shard_layout import Split, split_target_axes
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessMode,
    AccessQuantity,
    AccessRelations,
    BoundaryAccess,
    OutputStorage,
    StorageEffectClaim,
    StorageLink,
    elements_of,
    factored_window,
    logical_axes_of,
    logical_coordinates,
    moves,
    placed_window,
    register_access_relation,
    update_destination,
    window_source,
)
from tilefoundry.visitor_registry.relation_build import identity_access


@register_op(name="cache_update")
class CacheUpdate(Op):
    """Cache-update HIR operation; see `spec hir § CacheUpdate`."""

    cache = ParamDef(kind="input", pattern=Tensor)
    cur_pos = ParamDef(kind="input", pattern=Tensor)
    s = ParamDef(kind="input", pattern=Tensor)
    new = ParamDef(kind="input", pattern=Tensor)


def _cache_update_storage(call: "Call", ctx) -> StorageEffectClaim | None:
    """The new rows are written into the cache; the result is that cache."""
    return update_destination(call, ctx, destination=0)


def _limit(cache: tuple, supplied: tuple) -> int | None:
    """The most rows one call may write: the fewer of what each side states."""
    stated = [
        value
        for value in (
            _static(cache[1] if len(cache) > 1 else None),
            _static(supplied[1] if len(supplied) > 1 else None),
        )
        if value is not None
    ]
    return min(stated) if stated else None


def _static(extent) -> int | None:
    """One extent as a number, when it is one."""
    return extent if isinstance(extent, int) and not isinstance(extent, bool) else None


def _rows(expr) -> object:
    """How many rows this update writes: the number, or the value that says it.

    ``s`` is a runtime scalar, so when it is not written down the answer is the
    operand itself -- a relation can carry it as a parameter and a reader can
    bind it, which is what keeps this a quantity instead of charging the whole
    cache for a handful of rows.
    """
    if isinstance(expr, Constant) and isinstance(expr.value, int):
        return int(expr.value)
    return expr


def _written(rows, per_row: int, limit: int | None) -> AccessQuantity:
    """How many elements the update writes, or the range ``s`` leaves it in.

    ``1 <= s <= new.len`` and ``cur_pos + s <= cache.len`` are this Op's own
    contract, so a written-down ``s`` outside it describes no program and is
    refused here rather than turned into a negative complement downstream.
    """
    if not isinstance(limit, int):
        raise ValueError("CacheUpdate: the rows it may write have no stated bound")
    if isinstance(rows, int):
        if not 1 <= rows <= limit:
            raise ValueError(
                f"CacheUpdate writes {rows} rows, and this call may write "
                f"between 1 and {limit}"
            )
        return AccessQuantity(rows * per_row, rows * per_row)
    return AccessQuantity(
        per_row,
        limit * per_row,
        "CacheUpdate writes between one row and the fewer of what new supplies "
        "and what the cache holds",
    )


@register_access_relation(CacheUpdate)
def _cache_update_access(call: "Call", ctx) -> AccessRelations:
    """The result is the cache with ``s`` rows replaced at ``cur_pos``.

    How many rows move is ``s`` and where they land is ``cur_pos``; only the
    first is a quantity, and it is the same one wherever the second points. The
    two controls are rank-0, so their own boundaries are plain identities.
    Everything is counted by logical axis rather than by position: the row axis
    keeps its extent while a batch split shrinks the rest, and reading position
    one as the rows makes a participant answer with whatever its layout put
    there.
    """
    local_cache = ctx.local_type_of(call.args[0])
    local_new = ctx.local_type_of(call.args[3])
    logical_cache = ctx.type_of(call.args[0])
    logical_new = ctx.type_of(call.args[3])

    cache = _by_logical_axis(local_cache, logical_cache)
    supplied = _by_logical_axis(local_new, logical_new)
    rows = _rows(call.args[2])
    start = (
        int(call.args[1].value)
        if isinstance(call.args[1], Constant) and isinstance(call.args[1].value, int)
        else call.args[1]
    )
    offsets, extents = factored_window(
        (0, start, *(0 for _ in cache[2:])),
        (cache[0], rows, *cache[2:]),
        local_cache,
        logical_cache,
    )
    held = elements_of(local_cache)
    per_row = held // cache[1] if isinstance(cache[1], int) and cache[1] else 0
    written = _written(rows, per_row, _limit(tuple(cache), tuple(supplied)))
    kept = AccessQuantity(
        held - written.upper, held - written.lower, written.provenance
    )
    complement, reached = placed_window(offsets, extents, len(local_cache.shape))
    preserve = StorageLink(
        kind="preserve", input=0, where=complement, quantity=kept
    )
    return AccessRelations(
        inputs=(
            BoundaryAccess(complement, kept, AccessMode.TRANSFER),
            moves(_scalar_access(ctx, call.args[1]), 1),
            moves(_scalar_access(ctx, call.args[2]), 1),
            BoundaryAccess(
                window_source(
                    (0, start, *(0 for _ in cache[2:])),
                    len(local_cache.shape),
                    local_new,
                    logical_new,
                    logical_coordinates(local_cache, logical_cache),
                    (None, rows),
                ),
                written,
            ),
        ),
        outputs=(
            BoundaryAccess(
                reached,
                written,
                AccessMode.WRITE,
                OutputStorage((preserve,)),
            ),
        ),
        storage_effect=_cache_update_storage(call, ctx),
    )


def _is_scalar(shape) -> bool:
    """A scalar tensor: rank 0, or every dim is the literal 1."""
    return all(
        (isinstance(d, int) and d == 1) or (isinstance(d, Constant) and d.value == 1) for d in shape
    )


@register_typeinfer(CacheUpdate)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    cache_ty = ctx.type_of(call.args[0])
    cur_ty = ctx.type_of(call.args[1])
    s_ty = ctx.type_of(call.args[2])
    new_ty = ctx.type_of(call.args[3])
    if len(cache_ty.shape) != 4 or len(new_ty.shape) != 4:
        ctx.error(call, "cache and new must be rank-4 [B, len, kv_heads, head_dim]")
    if cache_ty.dtype != new_ty.dtype:
        ctx.error(call, f"cache/new dtype mismatch {cache_ty.dtype} vs {new_ty.dtype}")
    require_matching_partial_state(ctx, call, cache_ty, new_ty, "cache", "new")
    for ax, label in ((0, "B"), (2, "kv_heads"), (3, "head_dim")):
        if cache_ty.shape[ax] != new_ty.shape[ax]:
            ctx.error(
                call,
                f"cache/new {label} mismatch: {cache_ty.shape[ax]} vs {new_ty.shape[ax]}",
            )
    for t, name in ((cur_ty, "cur_pos"), (s_ty, "s")):
        if t.dtype != DType.i32:
            ctx.error(call, f"{name} must be an i32 scalar, got dtype {t.dtype}")
        if not _is_scalar(t.shape):
            ctx.error(call, f"{name} must be a scalar, got shape {t.shape}")
    cap, s_cap = cache_ty.shape[1], new_ty.shape[1]
    if isinstance(cap, int) and isinstance(s_cap, int) and s_cap > cap:
        ctx.error(call, f"S_CAP {s_cap} exceeds cache capacity {cap}")
    _reject_split_rows(ctx, call, cache_ty, new_ty)
    return cache_ty


def _divided_axes(type_) -> set[int]:
    """Which logical axes a layout really hands out in slices.

    A `Split` over a mesh axis of one gives the whole axis to the only
    participant there is, so it divides nothing.
    """
    layout = shard_layout_of(type_.layout)
    if layout is None:
        return set()
    targets = split_target_axes(layout, type_.shape)
    mesh = layout.mesh.layout.shape if layout.mesh is not None else ()
    return {
        targets[mesh_axis]
        for mesh_axis, attr in enumerate(layout.attrs)
        if isinstance(attr, Split)
        and mesh_axis < len(mesh)
        and mesh[mesh_axis] > 1
        and targets[mesh_axis] is not None
    }


def _reject_split_rows(ctx, call: "Call", cache_ty, new_ty) -> None:
    """Refuse an update whose two sides do not own the same thing.

    `cur_pos` is stated against the whole row axis, so a participant holding a
    slice of it would need its own offset to say which of its rows the update
    covers. Both sides are asked, because splitting the rows of `new` while the
    cache stays whole is the same question from the other end.

    Every other axis may be split, and the corpus does split the batch -- but
    only when both sides agree on it. One side sharded and the other not means
    a participant writes rows it does not hold, or holds rows nobody wrote.
    """
    held, supplied = _divided_axes(cache_ty), _divided_axes(new_ty)
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


@register_eval(CacheUpdate)
def _eval_cache_update(ctx):
    cache = ctx.args[0].data
    cur_pos = int(ctx.args[1].data.reshape(-1)[0].item())
    s = int(ctx.args[2].data.reshape(-1)[0].item())
    new = ctx.args[3].data
    capacity, s_cap = cache.shape[1], new.shape[1]
    if cur_pos < 0:
        raise EvalError(f"cache_update: cur_pos {cur_pos} must be >= 0")
    if not (1 <= s <= s_cap):
        raise EvalError(f"cache_update: s {s} must satisfy 1 <= s <= {s_cap}")
    if cur_pos + s > capacity:
        raise EvalError(
            f"cache_update: cur_pos + s ({cur_pos + s}) exceeds cache capacity {capacity}"
        )
    out = cache.clone()
    out[:, cur_pos : cur_pos + s] = new[:, :s].to(out.dtype)
    return TensorValue(data=out, type=ctx.result_type)


__all__ = ["CacheUpdate"]


def _scalar_access(ctx, arg) -> "isl.multi_aff":
    """A scalar operand, at whatever rank its own view gives it.

    A rank-0 value can arrive as a rank-1 position under a layout, and an image
    that names no coordinate cannot be composed with one that has a position.
    """
    held = ctx.local_type_of(arg)
    return identity_access(len(held.shape) if hasattr(held, "shape") else 0)


def _by_logical_axis(local, logical) -> list:
    """One extent per logical axis of a value, folding the positions it factored."""
    extents = [1] * len(logical.shape)
    for position, owner in enumerate(logical_axes_of(local, logical)):
        extents[owner] *= local.shape[position]
    return extents
