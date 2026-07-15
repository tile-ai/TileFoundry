"""MoE Routing primitive (K12 + K13).

SGLang K12 (moe_align_block_size) + K13 (count_and_sort_expert_tokens) produce
the routing metadata: a permuted token index list, per-expert offsets, and
per-expert token counts. We model both kernels as a single domain-specific
``MoERoute`` op rather than a generic ``Sort``.

"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.shard.shard_layout import partial_reductions
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelations,
    register_access_relation,
)


@register_op(name="moe_route")
class MoERoute(Op):
    """Multi-output (sorted_token_ids, expert_offsets, num_tokens_per_expert)."""
    topk_ids = ParamDef(kind="input", pattern=Tensor)
    num_experts = ParamDef(kind="attribute", annotation=int)
    block_size = ParamDef(kind="attribute", annotation=int, default=64)
@register_typeinfer(MoERoute)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    ids_ty = ctx.type_of(call.args[0])
    if len(ids_ty.shape) < 1:
        raise TypeError("MoERoute: topk_ids must be at least rank-1")
    if partial_reductions(ids_ty.layout):
        raise TypeError(
            "MoERoute: Partial input on topk_ids is unsound (routing is an "
            "opaque, data-dependent sort/permutation that does not commute "
            "with any reduction) — insert reshard(topk_ids, Broadcast) "
            "before this consumer"
        )
    # sorted_token_ids: leading-dim aligned, but actual length is data-dependent.
    # We reuse topk_ids.shape as a placeholder upper bound; downstream consumers
    # treat the routing handle as opaque.
    sorted_ty = TensorType(
        shape=ids_ty.shape,
        dtype=DType.i32,
        layout=ids_ty.layout,
        storage=ids_ty.storage,
    )
    offsets_ty = TensorType(
        shape=(call.target.num_experts + 1,),
        dtype=DType.i32,
        layout=ids_ty.layout,
        storage=ids_ty.storage,
    )
    counts_ty = TensorType(
        shape=(call.target.num_experts,),
        dtype=DType.i32,
        layout=ids_ty.layout,
        storage=ids_ty.storage,
    )
    return TupleType(fields=(sorted_ty, offsets_ty, counts_ty))

@register_access_relation(MoERoute)
def _moe_route_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL: routing is a runtime-dependent permutation; not affine. All
    boundaries return OPAQUE."""
    return AccessRelations(inputs=(OPAQUE,), outputs=(OPAQUE, OPAQUE, OPAQUE))

__all__ = ["MoERoute"]
