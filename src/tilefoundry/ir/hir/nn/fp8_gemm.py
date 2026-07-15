"""DeepGEMM-style FP8 block-scaled GEMM op.

Drawio K02 (QKV projection) and K08 (O projection) call SGLang's DeepGEMM
kernel: it consumes a quantized activation ``(x_fp8, x_scale)`` and a
quantized weight ``(w_fp8, w_scale)``, dequantizes internally, and produces a
``bf16`` output directly. Modeling this as a dedicated op (separate from the
plain bf16 ``MatMul`` used by K10 gate-router and H2 LM head) gives each
kernel a single op-class boundary with its own access_relation handler.

Signature (mirrors ``MatMul`` but with explicit per-block scale operands):

- ``lhs``       : ``[..., M, K]`` fp8e4m3
- ``lhs_scale`` : ``[..., M, K // group]`` f32
- ``rhs``       : ``[..., K, N]`` fp8e4m3
- ``rhs_scale`` : ``[..., K // group, N]`` f32

Output: ``[..., M, N]`` with dtype = ``out_dtype`` attribute (default bf16).
Layout / storage are inherited from ``lhs``.
"""
from __future__ import annotations

import isl

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.shard_layout import partial_reductions
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    register_access_relation,
)


@register_op(name="fp8_gemm")
class FP8GEMM(Op):
    """Block-scaled FP8 GEMM (DeepGEMM-style): fp8 × fp8 → bf16.

    Block size is the K-dim quantization group used by ``Quant`` (default 128,
    matching SGLang's ``per_token_group_quant_8bit_kernel``).
    """
    lhs = ParamDef(kind="input", pattern=Tensor)
    lhs_scale = ParamDef(kind="input", pattern=Tensor)
    rhs = ParamDef(kind="input", pattern=Tensor)
    rhs_scale = ParamDef(kind="input", pattern=Tensor)
    block = ParamDef(kind="attribute", annotation=int, default=128)
    out_dtype = ParamDef(kind="attribute", annotation=DType, default=DType.bf16)
@register_typeinfer(FP8GEMM)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    lhs = ctx.type_of(call.args[0])
    lhs_s = ctx.type_of(call.args[1])
    rhs = ctx.type_of(call.args[2])
    rhs_s = ctx.type_of(call.args[3])
    op = call.target
    if lhs.dtype != DType.fp8e4m3:
        ctx.error(call, f"FP8GEMM: lhs must be fp8e4m3, got {lhs.dtype}")
    if rhs.dtype != DType.fp8e4m3:
        ctx.error(call, f"FP8GEMM: rhs must be fp8e4m3, got {rhs.dtype}")
    if lhs_s.dtype != DType.f32:
        ctx.error(call, f"FP8GEMM: lhs_scale must be f32, got {lhs_s.dtype}")
    if rhs_s.dtype != DType.f32:
        ctx.error(call, f"FP8GEMM: rhs_scale must be f32, got {rhs_s.dtype}")
    if len(lhs.shape) < 2 or len(rhs.shape) < 2:
        ctx.error(call, "FP8GEMM requires rank >= 2 on both operands")

    batch = lhs.shape[:-2]
    if rhs.shape[:-2] != batch:
        ctx.error(call, f"FP8GEMM batch-dim mismatch {lhs.shape[:-2]} vs {rhs.shape[:-2]}")
    m = lhs.shape[-2]
    k = lhs.shape[-1]
    if rhs.shape[-2] != k:
        ctx.error(call, f"FP8GEMM K mismatch: lhs K={k} vs rhs K={rhs.shape[-2]}")
    n = rhs.shape[-1]

    # Block-scale shape: lhs_s == (..., M, K//block); rhs_s == (..., K//block, N).
    block = op.block
    if isinstance(k, int) and k % block != 0:
        ctx.error(call, f"FP8GEMM: K={k} not divisible by block={block}")
    expected_lhs_s_last = (k // block) if isinstance(k, int) else k
    expected_rhs_s_inner = (k // block) if isinstance(k, int) else k
    if lhs_s.shape[:-2] != batch or lhs_s.shape[-2] != m or lhs_s.shape[-1] != expected_lhs_s_last:
        ctx.error(
            call,
            f"FP8GEMM: lhs_scale shape {lhs_s.shape} != (..., M={m}, K/block={expected_lhs_s_last})",
        )
    if rhs_s.shape[:-2] != batch or rhs_s.shape[-2] != expected_rhs_s_inner or rhs_s.shape[-1] != n:
        ctx.error(
            call,
            f"FP8GEMM: rhs_scale shape {rhs_s.shape} != (..., K/block={expected_rhs_s_inner}, N={n})",
        )

    # A pre-existing Partial(reduction) on either operand (weight replication)
    # propagates only for "sum" — a block-scaled GEMM is linear in each
    # operand for the other fixed, but does not preserve order, so max/min
    # never commute.
    for arg, t in (("lhs", lhs), ("rhs", rhs)):
        bad = partial_reductions(t.layout) - {"sum"}
        if bad:
            ctx.error(
                call,
                f"FP8GEMM: Partial({sorted(bad)}) input on {arg} is unsound "
                f"(GEMM is linear, commutes with sum only) — insert "
                f"reshard({arg}, Broadcast) before this consumer",
            )

    return TensorType(
        shape=batch + (m, n),
        dtype=op.out_dtype,
        layout=lhs.layout,
        storage=lhs.storage,
    )

# ── Access relation (GLOBAL black-box) ────────────────────────────────

def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")

@register_access_relation(FP8GEMM)
def _fp8_gemm_access_relation(call: "Call", ctx) -> AccessRelations:
    """GLOBAL black-box: identity multi_aff per operand.

    K-dim reduction and per-block scale broadcast are internal kernel
    details, deferred to a future stage that lowers FP8GEMM to a more
    granular polyhedral form.
    """
    lhs_rank = len(ctx.type_of(call.args[0]).shape)
    lhs_s_rank = len(ctx.type_of(call.args[1]).shape)
    rhs_rank = len(ctx.type_of(call.args[2]).shape)
    rhs_s_rank = len(ctx.type_of(call.args[3]).shape)
    out_rank = len(ctx.type_of(call).shape)
    return AccessRelations(
        inputs=(
            _identity(lhs_rank),
            _identity(lhs_s_rank),
            _identity(rhs_rank),
            _identity(rhs_s_rank),
        ),
        outputs=(_identity(out_rank),),
    )

__all__ = ["FP8GEMM"]
