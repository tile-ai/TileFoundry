"""MoEExpertCompute big op (drawio K14-K19).

Black-box op covering FP8 quant of MoE input, fused per-expert up+gate GEMM,
SiLU(act_and_mul), FP8 quant of activation, fused per-expert down GEMM, and
weighted expert sum. Expert axis is a sparse distributed dimension — the
op carries no for-loop control flow; per-expert parallelism is expressed by
the ``ShardLayout`` of ``w_*`` weights (out of scope for this baseline round).

"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import partial_reductions
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelations,
    register_access_relation,
)


@register_op(name="moe_expert_compute")
class MoEExpertCompute(Op):
    """K14-K19 absorbed into one black-box. Output shape == x.shape."""
    x = ParamDef(kind="input", pattern=Tensor)
    topk_ids = ParamDef(kind="input", pattern=Tensor)
    topk_weights = ParamDef(kind="input", pattern=Tensor)
    routing = ParamDef(kind="input", pattern=Tensor)
    w_gate = ParamDef(kind="input", pattern=Tensor)
    w_gate_scale = ParamDef(kind="input", pattern=Tensor)
    w_up = ParamDef(kind="input", pattern=Tensor)
    w_up_scale = ParamDef(kind="input", pattern=Tensor)
    w_down = ParamDef(kind="input", pattern=Tensor)
    w_down_scale = ParamDef(kind="input", pattern=Tensor)
    num_experts = ParamDef(kind="attribute", annotation=int, default=0)
    topk = ParamDef(kind="attribute", annotation=int, default=0)
    intermediate = ParamDef(kind="attribute", annotation=int, default=0)
    quant_scheme = ParamDef(kind="attribute", annotation=str, default="per_token_group")
    quant_group = ParamDef(kind="attribute", annotation=int, default=128)
@register_typeinfer(MoEExpertCompute)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        raise TypeError("MoEExpertCompute: x must be at least rank-1")
    if partial_reductions(x_ty.layout):
        raise TypeError(
            "MoEExpertCompute: Partial input on x is unsound (an opaque "
            "black box with nonlinear internals — SiLU, quant, weighted "
            "expert sum — does not commute with any reduction) — insert "
            "reshard(x, Broadcast) before this consumer"
        )
    return TensorType(
        shape=x_ty.shape,
        dtype=x_ty.dtype,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )

@register_access_relation(MoEExpertCompute)
def _moe_expert_compute_access_relation(
    call: "Call", ctx: "TypeInferContext"
) -> AccessRelations:
    """Big op — all boundaries OPAQUE per D12."""
    n_in = len(call.args)
    return AccessRelations(
        inputs=tuple(OPAQUE for _ in range(n_in)),
        outputs=(OPAQUE,),
    )

__all__ = ["MoEExpertCompute"]
