"""Functional KV-cache update HIR primitive.

Returns a new cache of the **same** (static) shape with ``new[:, :s]`` written at
``cache[:, cur_pos : cur_pos + s]`` and all other positions unchanged. The write
region (``cur_pos`` / ``s``) is runtime scalar data, never a shape dim, so the
cache shape stays static (no compound ``DimVar`` context axis). It is a pure
value-form op; an in-place realization is a lowering concern (the output is
anchored on the input cache buffer).
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

# Data-dependent write region (``cur_pos`` / ``s`` are runtime values), so no
# affine access relation is registered — the boundaries are opaque.


@register_op(name="cache_update")
class CacheUpdate(Op):
    """Write ``new[:, :s]`` into ``cache[:, cur_pos : cur_pos + s]``; return the
    updated cache (same shape). ``cur_pos`` / ``s`` are runtime scalar tensors."""
    cache = ParamDef(kind="input", pattern=Tensor)
    cur_pos = ParamDef(kind="input", pattern=Tensor)
    s = ParamDef(kind="input", pattern=Tensor)
    new = ParamDef(kind="input", pattern=Tensor)


def _is_scalar(shape) -> bool:
    """A scalar tensor: rank 0, or every dim is the literal 1."""
    return all(
        (isinstance(d, int) and d == 1) or (isinstance(d, Constant) and d.value == 1)
        for d in shape
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
