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
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessQuantity,
    AccessRelations,
    BoundaryAccess,
    OperandValue,
    StorageEffectClaim,
    WindowAccess,
    elements_of,
    moves,
    register_access_relation,
    update_destination,
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


def _rows(expr, capacity, supplied) -> object:
    """How many rows this update writes: the number, or the value that says it.

    ``s`` is a runtime scalar, and what it may be is the Op's own contract: at
    least one row, and no more than either the cache holds or ``new`` brought.
    That range is what keeps a quantity a quantity instead of charging the whole
    cache for a handful of rows.
    """
    if isinstance(expr, Constant) and isinstance(expr.value, int):
        return int(expr.value)
    limits = [value for value in (_static(capacity), _static(supplied)) if value is not None]
    return OperandValue(operand=2, bound=(1, min(limits)) if limits else None)


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
    """
    cache = tuple(ctx.local_type_of(call.args[0]).shape)
    supplied = tuple(ctx.local_type_of(call.args[3]).shape)
    rows = _rows(
        call.args[2],
        cache[1] if len(cache) > 1 else None,
        supplied[1] if len(supplied) > 1 else None,
    )
    extents = (cache[0], rows, *cache[2:])
    start = (
        int(call.args[1].value)
        if isinstance(call.args[1], Constant) and isinstance(call.args[1].value, int)
        else OperandValue(operand=1)
    )
    offsets = (0, start, *(0 for _ in cache[2:]))
    held = elements_of(ctx.local_type_of(call.args[0]))
    per_row = held // cache[1] if isinstance(cache[1], int) and cache[1] else 0
    written = _written(rows, per_row, _limit(cache, supplied))
    kept = AccessQuantity(
        held - written.upper, held - written.lower, written.provenance
    )
    return AccessRelations(
        inputs=(
            BoundaryAccess(WindowAccess(offsets, extents, complement=True), kept),
            moves(identity_access(0), 1),
            moves(identity_access(0), 1),
            BoundaryAccess(WindowAccess(tuple(0 for _ in cache), extents), written),
        ),
        outputs=(BoundaryAccess(identity_access(len(cache)), written),),
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
    return cache_ty


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
