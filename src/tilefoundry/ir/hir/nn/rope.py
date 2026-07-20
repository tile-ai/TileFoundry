"""Rotary Position Embedding (RoPE) HIR primitive.

SGLang baseline kernel K04. Applies position-dependent rotation to Q and K
along the last (head_dim) axis, using precomputed cos/sin caches indexed by
``pos_ids``.


Multi-output op: returns a tuple ``(q_rope, k_rope)``. Both share input shape /
dtype / layout / storage with their respective Q / K input.
"""
from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TupleType
from tilefoundry.ir.types.shard.shard_layout import Broadcast, ShardLayout
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelations,
    register_access_relation,
)
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op
class RoPE(Op):
    """Rotary position embedding on Q and K. ``head_dim`` must be even."""
    q = ParamDef(kind="input", pattern=Tensor)
    k = ParamDef(kind="input", pattern=Tensor)
    cos_cache = ParamDef(kind="input", pattern=Tensor)
    sin_cache = ParamDef(kind="input", pattern=Tensor)
    pos_ids = ParamDef(kind="input", pattern=Tensor)


def _is_replicated_at(layout, axis: int) -> bool:
    if not isinstance(layout, ShardLayout) or axis >= len(layout.attrs):
        return True
    return isinstance(layout.attrs[axis], Broadcast)


def _check_linear_inputs(operands) -> None:
    states = {
        name: partial_reductions_by_axis(ty.layout) for name, ty in operands
    }
    for axis in range(max((len(s) for s in states.values()), default=0)):
        partials = [
            (name, reductions[axis])
            for name, reductions in states.items()
            if axis < len(reductions) and reductions[axis] is not None
        ]
        if not partials:
            continue
        for name, reduction in partials:
            if name != operands[0][0]:
                raise TypeError(
                    f"RoPE: {name} carries Partial({reduction}) on mesh axis "
                    f"{axis}; the output tuple cannot preserve this secondary "
                    f"state. Use Reshard({name}, Broadcast) before this consumer"
                )
        if len(partials) > 1:
            details = ", ".join(
                f"{name}=Partial({reduction})" for name, reduction in partials
            )
            raise TypeError(
                f"RoPE: multiple value-carrying Partials on mesh axis {axis} "
                f"({details}) do not commute; insert Reshard to Broadcast "
                "before this consumer"
            )
        name, reduction = partials[0]
        if reduction != "sum":
            raise TypeError(
                f"RoPE: {name} carries Partial({reduction}) on mesh axis "
                f"{axis}; RoPE commutes with sum only. Insert reshard({name}, "
                "Broadcast) before this consumer"
            )
        for other_name, other_ty in operands:
            if other_name == name:
                continue
            if not _is_replicated_at(other_ty.layout, axis):
                raise TypeError(
                    f"RoPE: {name} carries Partial(sum) on mesh axis {axis}, "
                    f"but {other_name} is not Broadcast/replicated on that "
                    f"axis; insert reshard({other_name}, Broadcast) before "
                    "this consumer"
                )
@register_typeinfer(RoPE)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    q_ty = ctx.type_of(call.args[0])
    k_ty = ctx.type_of(call.args[1])
    cos_ty = ctx.type_of(call.args[2])
    sin_ty = ctx.type_of(call.args[3])
    pos_ty = ctx.type_of(call.args[4])
    if not q_ty.shape or not k_ty.shape:
        raise TypeError("RoPE: q and k must be at least rank-1")
    head_dim_q = q_ty.shape[-1]
    head_dim_k = k_ty.shape[-1]
    if isinstance(head_dim_q, int) and head_dim_q % 2 != 0:
        raise TypeError(f"RoPE: q head_dim {head_dim_q} must be even")
    if isinstance(head_dim_k, int) and head_dim_k % 2 != 0:
        raise TypeError(f"RoPE: k head_dim {head_dim_k} must be even")
    if (
        isinstance(head_dim_q, int)
        and isinstance(head_dim_k, int)
        and head_dim_q != head_dim_k
    ):
        raise TypeError(
            f"RoPE: q head_dim {head_dim_q} != k head_dim {head_dim_k}"
        )
    # q*cos + rotate_half(q)*sin is multilinear in each value input. Each
    # output branch therefore allows one sum Partial only when all other
    # value-carrying inputs on that mesh axis are replicated.
    _check_linear_inputs(
        (("q", q_ty), ("cos_cache", cos_ty), ("sin_cache", sin_ty)),
    )
    _check_linear_inputs(
        (("k", k_ty), ("cos_cache", cos_ty), ("sin_cache", sin_ty)),
    )
    for axis, reduction in enumerate(partial_reductions_by_axis(pos_ty.layout)):
        if reduction is not None:
            raise TypeError(
                f"RoPE: pos_ids carries Partial({reduction}) on mesh axis "
                f"{axis}; indexed cache access does not commute. Insert "
                "reshard(pos_ids, Broadcast) before this consumer"
            )
    return TupleType(fields=(q_ty, k_ty))

@register_access_relation(RoPE)
def _rope_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL level.

    Inputs:
      - q, k: per-element identity (rotation is per (token, head, head_dim/2 pair))
      - cos_cache, sin_cache: indexed by pos_ids → opaque (data-dependent index)
      - pos_ids: opaque (1D index input feeding cache lookup)

    Outputs:
      - q_rope, k_rope: per-element identity vs Q / K respectively.
    """
    q_ty = ctx.type_of(call.args[0])
    k_ty = ctx.type_of(call.args[1])

    def _ident(rank: int) -> "isl.multi_aff":
        dims = ", ".join(f"i{i}" for i in range(rank))
        return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")

    q_id = _ident(len(q_ty.shape))
    k_id = _ident(len(k_ty.shape))

    return AccessRelations(
        inputs=(q_id, k_id, OPAQUE, OPAQUE, OPAQUE),
        outputs=(q_id, k_id),
    )

@register_eval(RoPE)
def _eval_rope(ctx):
    # Layout is [batch, seq, head, head_dim]: cos/sin are gathered per token
    # from the caches by ``pos_ids`` and broadcast over the batch and head axes.
    # The rotation is the rotate-half form q*cos + rotate_half(q)*sin.
    q = ctx.args[0].data.float()
    k = ctx.args[1].data.float()
    pos = ctx.args[4].data.reshape(-1).long()
    cos = ctx.args[2].data[pos].float()[None, :, None, :]
    sin = ctx.args[3].data[pos].float()[None, :, None, :]

    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return TupleValue(
        elements=(
            TensorValue(
                data=q_out.to(to_torch_dtype(ctx.result_type.fields[0].dtype)),
                type=ctx.result_type.fields[0],
            ),
            TensorValue(
                data=k_out.to(to_torch_dtype(ctx.result_type.fields[1].dtype)),
                type=ctx.result_type.fields[1],
            ),
        )
    )


__all__ = ["RoPE"]
