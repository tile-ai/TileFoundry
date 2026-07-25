"""Runtime twin of the DeepSeek-V4-Flash causal-LM tree: same nodes, same
entry names, same function/child names as ``causal_lm.py`` / ``attention.py``
/ ``moe.py`` (verified one-to-one by ``@runtime_module`` at decoration time),
with every ``@func`` leaf replaced by a hand-written torch (or, for
``residual_add``, real CUDA) kernel body. Orchestration (``forward`` /
``init_caches`` / ``prepare_inputs_for_generation``) is never rewritten here
-- it is the semantic module's own methods, reused verbatim through
``self.<fn>`` / ``self.<child>`` resolving to this twin's kernels/children
(``src/tilefoundry/runtime/decorator.py``'s ``_Twin.forward``).

Four ``@runtime_module`` classes, one per semantic node:

    DeepseekV4ForCausalLMRT   embed, final_rms_norm, lm_head        (+ layerN children)
    └─ DeepseekV4DecoderLayerRT   pre_attn_rms_norm, pre_moe_rms_norm, residual_add
       ├─ DeepseekV4AttentionRT      mla_kv_update, mla_attend
       └─ DeepseekV4MoERT            shared_fp8_dequant_w1/w2, moe_experts_core,
                                      moe_hash_gather, shared_expert,
                                      combine_expert_outputs, deepseek_v4_flash_moe_hash

``build_runtime_causal_lm(config, ir)`` is the one seam callers need: *ir* is
the already-built semantic root (``causal_lm.build_causal_lm(config)``); its
own nested tree (``ir.modules[0]``, that node's ``.attention`` / ``.moe``)
supplies the per-node semantic ``sem`` each ``@runtime_module(sem)`` call
below validates against, and ``ir.modules`` (however many layers *config*
built) drives how many ``layerN`` child attributes the root class gets --
so this works unchanged at ``DSV4Config.tiny()`` (1 layer) or ``REAL`` (43).

Precision discipline (this is the actual gate: bf16 rel_l2 <= 1e-3 punishes a
single mismatched upcast): every kernel below mirrors its ``@func``'s exact
per-op dtype handling, copied from the evaluator handlers under
``src/tilefoundry/ir/hir/{nn,tensor,math}/*.py``, not "mathematically
equivalent" torch -- ``matmul`` stays in the operands' own dtype (no upcast)
*except* ``moe_hash_gather``'s routing-gate matmul, which the semantic body
itself upcasts to f32 before multiplying (mirrored here, not "corrected"
away); ``rms_norm`` upcasts x and weight to f32 and uses eps=1e-6 (the DSL
op's own default -- none of the semantic bodies override it, so this
hardcodes the same default rather than threading ``config.rms_eps``, which
would silently diverge from what the reference actually runs); ``softmax``
reduces in f32 and casts back; the interleaved-pairs RoPE and the KV/expert
fp8 block dequant upcast to f32 exactly where the semantic body's own
``tf.cast(..., dtype="f32")`` calls do, and nowhere else.

The KV cache write is functional, matching ``tf.cache_update``'s evaluator
(``ir/hir/tensor/cache_update.py``: ``out = cache.clone(); out[:, cur_pos:cur_pos+s]
= new[:, :s]; return out``) exactly: ``mla_kv_update`` never mutates its
``kv_cache0`` argument, it returns a new tensor.
"""
from __future__ import annotations

import torch

from tests.models.deepseek_v4_flash.config import (
    FP8E4M3_MAX,
    FP8E4M3_QUANT_EPS,
    KV_QUANT_BLOCK,
    DSV4Config,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.runtime import RuntimeModule, runtime_func, runtime_module

_BF16 = torch.bfloat16
_F32 = torch.float32
_FP8E4M3 = torch.float8_e4m3fn

# ───────────────────────────── shared helpers ──────────────────────────────
# Plain, undecorated functions -- not kernels themselves, called *from* kernel
# bodies below (mirrors "Helpers must be plain undecorated methods").


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x * rsqrt(mean(x**2, -1, keepdim) + eps) * weight``, f32 internally,
    cast back to x's dtype -- ``ir/hir/nn/rms_norm.py``'s evaluator, verbatim."""
    xf = x.float()
    wf = weight.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(ms + eps) * wf
    return out.to(x.dtype)


def _gather_axis0(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``tf.gather(x, indices, axis=0)`` (``batch_dims=0``): index_select the
    flattened indices, reshape to ``indices.shape + x.shape[1:]`` --
    ``ir/hir/tensor/gather.py``'s evaluator, non-batched branch."""
    idx = indices.reshape(-1).long()
    out = torch.index_select(x, 0, idx)
    return out.reshape(tuple(indices.shape) + tuple(x.shape[1:]))


def _block_dequant(
    weight_fp8: torch.Tensor, scale: torch.Tensor, quant_block: int, out_shape: tuple[int, int],
) -> torch.Tensor:
    """Block dequant for a plain 2-D ``(rows, cols)`` weight: reshape into
    ``(row_blocks, quant_block, col_blocks, quant_block)`` tiles, multiply by
    the scale broadcast per tile, reshape back. Both operands cast to bf16
    before the multiply (no f32 upcast) -- ``moe.py``'s
    ``shared_fp8_dequant_w1`` / ``w2`` bodies, verbatim."""
    rows, cols = out_shape
    row_blocks, col_blocks = rows // quant_block, cols // quant_block
    blocks = weight_fp8.to(_BF16).reshape(row_blocks, quant_block, col_blocks, quant_block)
    block_scale = scale.to(_BF16).reshape(row_blocks, 1, col_blocks, 1)
    return (blocks * block_scale).reshape(rows, cols)


# ───────────────────────── CUDA kernel: residual_add ───────────────────────
# The one real CUDA kernel (torch.utils.cpp_extension.load_inline), proving
# the C++/FFI path a hand-written kernel actually takes. residual_add is the
# smallest @func in the tree (plain elementwise add), so nvcc stays fast.
# Compiled lazily (first call, not import time) and cached at module scope --
# one compile, shared by every DeepseekV4DecoderLayerRT instance/layer.

_RESIDUAL_ADD_EXT_NAME = "tilefoundry_dsv4_flash_residual_add"
_residual_add_ext = None

_RESIDUAL_ADD_CPP_SRC = "torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b);"

_RESIDUAL_ADD_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void residual_add_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hadd(a[i], b[i]);
    }
}

torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "residual_add_cuda: inputs must be CUDA tensors");
    TORCH_CHECK(
        a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
        "residual_add_cuda: inputs must be bf16"
    );
    TORCH_CHECK(a.sizes() == b.sizes(), "residual_add_cuda: shape mismatch");
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = torch::empty_like(a_c);
    const int64_t n = a_c.numel();
    const int threads = 128;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    residual_add_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(b_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        n
    );
    return out;
}
"""


def _get_residual_add_ext():
    global _residual_add_ext
    if _residual_add_ext is None:
        from torch.utils.cpp_extension import load_inline  # noqa: PLC0415 -- lazy, heavy (nvcc)

        _residual_add_ext = load_inline(
            name=_RESIDUAL_ADD_EXT_NAME,
            cpp_sources=_RESIDUAL_ADD_CPP_SRC,
            cuda_sources=_RESIDUAL_ADD_CUDA_SRC,
            functions=["residual_add_cuda"],
            verbose=False,
        )
    return _residual_add_ext


def _residual_add_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _get_residual_add_ext().residual_add_cuda(a, b)


# ──────────────────────────────── attention ────────────────────────────────


def _build_attention_rt(config: DSV4Config, sem: Module) -> type:
    head_dim = config.head_dim
    nope_dim = config.nope_dim
    rope_dim = config.rope_dim
    kv_quant_blocks = config.kv_quant_blocks
    n_heads = config.n_heads
    window = config.window
    o_groups = config.o_groups
    o_lora_rank = config.o_lora_rank
    wo_a_in = config.wo_a_in
    wo_a_out = config.wo_a_out
    q_proj = config.q_proj

    @runtime_module(sem)
    class DeepseekV4AttentionRT:
        @runtime_func
        def mla_kv_update(self, hidden, gamma_kv, w_kv, cos_pos, sin_pos, kv_cache0, cur_pos, s):
            kv = torch.matmul(hidden, w_kv)
            kv_n = _rms_norm(kv, gamma_kv)
            kv_4d = kv_n.reshape(1, 1, 1, head_dim)
            kv_nope = kv_4d[..., :nope_dim]
            kv_rope_in = kv_4d[..., nope_dim:]

            # fp8 fake-quant of the non-rope portion: block absmax -> clamp to
            # a floor -> round the scale up to a power of two -> divide ->
            # clamp to the e4m3 grid -> real fp8 round trip -> dequant. All in
            # f32, matching the semantic body's own explicit cast chain.
            kv_nope_f32 = kv_nope.float()
            kv_nope_blk = kv_nope_f32.reshape(1, 1, 1, kv_quant_blocks, KV_QUANT_BLOCK)
            kv_amax = kv_nope_blk.abs().amax(dim=-1, keepdim=True).clamp_min(FP8E4M3_QUANT_EPS)
            kv_scale = torch.exp2(torch.ceil(torch.log2(kv_amax / FP8E4M3_MAX)))
            kv_scaled = (kv_nope_blk / kv_scale).clamp(-FP8E4M3_MAX, FP8E4M3_MAX)
            kv_q_fp8 = kv_scaled.to(_FP8E4M3)
            kv_dq = kv_q_fp8.to(_F32) * kv_scale
            kv_nope_q = kv_dq.reshape(1, 1, 1, nope_dim).to(_BF16)

            # Interleaved-pairs RoPE on the rope slice: f32 upcast for the
            # rotation, single rounding back to bf16 at the end.
            kv_r0, kv_r1 = kv_rope_in[..., 0::2], kv_rope_in[..., 1::2]
            kv_r0f, kv_r1f = kv_r0.float(), kv_r1.float()
            kv_o0 = (kv_r0f * cos_pos - kv_r1f * sin_pos).to(_BF16)
            kv_o1 = (kv_r0f * sin_pos + kv_r1f * cos_pos).to(_BF16)
            kv_rope_out = torch.stack((kv_o0, kv_o1), dim=-1).reshape(1, 1, 1, rope_dim)
            kv_final = torch.cat((kv_nope_q, kv_rope_out), dim=-1)

            # Functional cache write (tf.cache_update): a NEW tensor, cache0
            # left untouched -- never `cache0[...] = ...` in place.
            cur_pos_i = int(cur_pos.reshape(-1)[0].item())
            s_i = int(s.reshape(-1)[0].item())
            out = kv_cache0.clone()
            out[:, cur_pos_i : cur_pos_i + s_i] = kv_final[:, :s_i].to(out.dtype)
            return out

        @runtime_func
        def mla_attend(
            self, hidden, gamma_q_lora, w_q_a, w_q_b, ones_head_dim, cos_pos, sin_pos,
            kv_cache, attn_mask, attn_sink, scale, w_o_a, w_o_b,
        ):
            # Low-rank Q, per-head unweighted RMS rescale (all-ones weight),
            # then the same interleaved-pairs RoPE as mla_kv_update.
            q_lat = _rms_norm(torch.matmul(hidden, w_q_a), gamma_q_lora)
            q_full = torch.matmul(q_lat, w_q_b)
            q = q_full.reshape(1, 1, n_heads, head_dim)
            q_rescaled = _rms_norm(q, ones_head_dim)
            q_nope = q_rescaled[..., :nope_dim]
            q_rope_in = q_rescaled[..., nope_dim:]

            q_r0, q_r1 = q_rope_in[..., 0::2], q_rope_in[..., 1::2]
            q_r0f, q_r1f = q_r0.float(), q_r1.float()
            q_o0 = (q_r0f * cos_pos - q_r1f * sin_pos).to(_BF16)
            q_o1 = (q_r0f * sin_pos + q_r1f * cos_pos).to(_BF16)
            q_rope_out = torch.stack((q_o0, q_o1), dim=-1).reshape(1, 1, n_heads, rope_dim)
            q_final = torch.cat((q_nope, q_rope_out), dim=-1)

            # MQA broadcast (n_kv_heads==1 -> n_heads); kv_cache is both K/V
            # (MLA-absorbed). matmuls stay in bf16 (no upcast) throughout.
            k_b = torch.repeat_interleave(kv_cache, n_heads, dim=2)
            q_h = q_final.permute(0, 2, 1, 3).contiguous()
            k_h = k_b.permute(0, 2, 1, 3).contiguous()
            q_s = q_h * scale
            k_t = k_h.permute(0, 1, 3, 2).contiguous()
            scores = torch.matmul(q_s, k_t) + attn_mask

            # attn_sink as an extra softmax column, sliced back off before P@V.
            scores_ext = torch.cat((scores, attn_sink), dim=-1)
            probs_ext = torch.softmax(scores_ext.float(), dim=-1).to(_BF16)
            probs = probs_ext[..., :window]
            ctx = torch.matmul(probs, k_h)

            # Inverse-RoPE the context (conjugate angle: cos+sin swap signs).
            ctx_nope = ctx[..., :nope_dim]
            ctx_rope_in = ctx[..., nope_dim:]
            ctx_r0, ctx_r1 = ctx_rope_in[..., 0::2], ctx_rope_in[..., 1::2]
            ctx_r0f, ctx_r1f = ctx_r0.float(), ctx_r1.float()
            ctx_o0 = (ctx_r0f * cos_pos + ctx_r1f * sin_pos).to(_BF16)
            ctx_o1 = (ctx_r1f * cos_pos - ctx_r0f * sin_pos).to(_BF16)
            ctx_rope_out = torch.stack((ctx_o0, ctx_o1), dim=-1).reshape(1, n_heads, 1, rope_dim)
            ctx_final = torch.cat((ctx_nope, ctx_rope_out), dim=-1)

            attn_out_heads_last = ctx_final.permute(0, 2, 1, 3).contiguous()
            o_flat = attn_out_heads_last.reshape(1, 1, q_proj)

            # Grouped low-rank O projection: one batched matmul over the
            # o_groups axis (config-driven, not an unrolled fixed count).
            o_grouped = o_flat.reshape(o_groups, 1, 1, wo_a_in)
            w_o_a_grouped = w_o_a.reshape(o_groups, 1, wo_a_in, o_lora_rank)
            y_grouped = torch.matmul(o_grouped, w_o_a_grouped)
            y = y_grouped.reshape(1, 1, wo_a_out)
            return torch.matmul(y, w_o_b)

    return DeepseekV4AttentionRT


# ────────────────────────────────── moe ────────────────────────────────────


def _build_moe_rt(config: DSV4Config, sem: Module) -> type:
    dim = config.dim
    moe_inter = config.moe_inter
    n_act = config.n_act
    route_scale = config.route_scale
    swiglu_limit = config.swiglu_limit
    quant_block = config.quant_block
    blk_dim = config.blocks(dim)
    blk_inter = config.blocks(moe_inter)

    @runtime_module(sem)
    class DeepseekV4MoERT:
        @runtime_func
        def shared_fp8_dequant_w1(self, weight, scale):
            return _block_dequant(weight, scale, quant_block, (moe_inter, dim))

        @runtime_func
        def shared_fp8_dequant_w2(self, weight, scale):
            return _block_dequant(weight, scale, quant_block, (dim, moe_inter))

        @runtime_func
        def moe_experts_core(
            self, x, gweights, eids, w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
        ):
            xt = x.reshape(1, dim)

            def _expert_weight(weight, scale, block_shape):
                b0, b1 = block_shape
                gw = _gather_axis0(weight, eids).to(_BF16)
                gs = _gather_axis0(scale, eids).to(_BF16)
                gw = gw.reshape(1, n_act, b0, quant_block, b1, quant_block)
                gs = gs.reshape(1, n_act, b0, 1, b1, 1)
                return (gw * gs).reshape(1, n_act, b0 * quant_block, b1 * quant_block)

            w1 = _expert_weight(w1_weight, w1_scale, (blk_inter, blk_dim))
            w3 = _expert_weight(w3_weight, w3_scale, (blk_inter, blk_dim))
            w2 = _expert_weight(w2_weight, w2_scale, (blk_dim, blk_inter))

            # Per-expert batched matmul (bf16, no upcast), swiglu with the
            # clamp limit (up: both sides; gate: upper side only), cast to f32
            # only where the semantic body's own tf.cast(..., "f32") sits.
            token = xt.reshape(1, 1, dim, 1)
            gate_value = torch.matmul(w1, token).reshape(1, n_act, moe_inter).float()
            up_value = torch.matmul(w3, token).reshape(1, n_act, moe_inter).float()
            up_value = up_value.clamp(-swiglu_limit, swiglu_limit)
            gate_value = torch.clamp(gate_value, max=swiglu_limit)
            hidden = (gate_value * torch.sigmoid(gate_value)) * up_value
            hidden = hidden.to(_BF16).reshape(1, n_act, moe_inter, 1)
            expert_output = torch.matmul(w2, hidden).reshape(1, n_act, dim).float()
            weighted = expert_output * gweights.reshape(1, n_act, 1)
            return weighted.to(_BF16)

        @runtime_func
        def moe_hash_gather(
            self, x, gate_weight, tid2eid, token_ids,
            w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
        ):
            xt = x.reshape(1, dim)
            # Deliberate exception to "matmul stays in bf16": the semantic
            # body itself casts both operands to f32 before this one, for
            # routing-score precision -- mirrored here, not "fixed" away.
            gate = torch.matmul(xt.float(), gate_weight.float().t())
            softplus = torch.log(torch.exp(gate) + 1.0)
            scores = softplus * torch.rsqrt(softplus)
            eids = _gather_axis0(tid2eid, token_ids)
            gweights = torch.gather(scores, 1, eids)
            weight_sum = gweights.sum(dim=-1, keepdim=True)
            gweights = (gweights / weight_sum) * route_scale
            # Sibling call: self.moe_experts_core auto-fills its own 6 consts
            # from _bound, exactly like the semantic body's bare-name call
            # passing them through explicitly -- same tensors, either way.
            return self.moe_experts_core(x, gweights, eids)

        @runtime_func
        def shared_expert(
            self, x, shared_w1_weight, shared_w1_scale, shared_w3_weight, shared_w3_scale,
            shared_w2_weight, shared_w2_scale,
        ):
            xt = x.reshape(1, dim)
            # w3 reuses the w1 dequant helper (same (moe_inter, dim) shape),
            # matching the semantic body's own reuse.
            w1 = self.shared_fp8_dequant_w1(shared_w1_weight, shared_w1_scale)
            w3 = self.shared_fp8_dequant_w1(shared_w3_weight, shared_w3_scale)
            gate = torch.matmul(xt, w1.t()).float()
            up = torch.matmul(xt, w3.t()).float()
            up = up.clamp(-swiglu_limit, swiglu_limit)
            gate = torch.clamp(gate, max=swiglu_limit)
            hidden = ((gate * torch.sigmoid(gate)) * up).to(_BF16)
            w2 = self.shared_fp8_dequant_w2(shared_w2_weight, shared_w2_scale)
            output = torch.matmul(hidden, w2.t()).to(_BF16)
            return output.reshape(1, 1, dim)

        @runtime_func
        def combine_expert_outputs(self, routed, shared):
            return routed + shared

        @runtime_func
        def deepseek_v4_flash_moe_hash(
            self, hidden, gate_weight, tid2eid, token_ids,
            w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
            shared_w1_weight, shared_w1_scale, shared_w3_weight, shared_w3_scale,
            shared_w2_weight, shared_w2_scale,
        ):
            # Every *_weight/*_scale param above is accepted (same signature
            # as the semantic @func) but only reached through the sibling
            # self.-calls below, which pull the identical bound tensors
            # themselves -- exactly how the semantic entry's own body only
            # forwards them, never touching them directly either.
            routed_experts = self.moe_hash_gather(hidden, token_ids)
            routed_reduced = routed_experts.sum(dim=1, keepdim=False)
            routed_value = routed_reduced.to(_BF16).reshape(1, 1, dim)
            shared_value = self.shared_expert(hidden)
            return self.combine_expert_outputs(routed_value, shared_value)

    return DeepseekV4MoERT


# ─────────────────────────────── decoder layer ─────────────────────────────


def _build_decoder_layer_rt(sem: Module, attention_cls: type, moe_cls: type) -> type:
    @runtime_module(sem)
    class DeepseekV4DecoderLayerRT:
        @runtime_func
        def pre_attn_rms_norm(self, x, pre_attn_norm_weight):
            return _rms_norm(x, pre_attn_norm_weight)

        @runtime_func
        def pre_moe_rms_norm(self, x, pre_moe_norm_weight):
            return _rms_norm(x, pre_moe_norm_weight)

        @runtime_func
        def residual_add(self, a, b):
            return _residual_add_cuda(a, b)

        attention = attention_cls
        moe = moe_cls

    return DeepseekV4DecoderLayerRT


# ────────────────────────────────── root ───────────────────────────────────


def _build_root_funcs(config: DSV4Config) -> dict[str, object]:
    dim = config.dim
    vocab = config.vocab

    @runtime_func
    def embed(self, table, token_ids):
        idx = token_ids.reshape(-1).long()
        return torch.index_select(table, 0, idx).reshape(1, 1, dim)

    @runtime_func
    def final_rms_norm(self, hidden, final_norm_weight):
        return _rms_norm(hidden, final_norm_weight)

    @runtime_func
    def lm_head(self, hidden, lm_head_weight):
        # lm_head_weight arrives already canonical (dim, vocab): the
        # transpose from the checkpoint's (vocab, dim) is prepare-time-only
        # (the semantic function's own .converter), never re-applied here.
        logits = torch.matmul(hidden.reshape(1, dim), lm_head_weight)
        return logits.reshape(1, 1, vocab)

    return {"embed": embed, "final_rms_norm": final_rms_norm, "lm_head": lm_head}


def build_runtime_causal_lm(config: DSV4Config, ir: Module) -> RuntimeModule:
    """The runtime twin of `build_causal_lm(config)`'s tree; `ir` is that
    semantic root (its children/entry/functions drive the twin)."""
    if not ir.modules:
        raise ValueError("build_runtime_causal_lm: ir has no decoder layers")
    layer0_ir = ir.modules[0]
    attention_ir = layer0_ir.attention
    moe_ir = layer0_ir.moe

    attention_cls = _build_attention_rt(config, attention_ir)
    moe_cls = _build_moe_rt(config, moe_ir)
    decoder_layer_cls = _build_decoder_layer_rt(layer0_ir, attention_cls, moe_cls)

    # One shared decoder-layer class, one attribute per real layer name --
    # config.n_layers many (1 at tiny(), 43 at REAL), read off ir.modules
    # itself rather than hardcoded.
    namespace: dict[str, object] = dict(_build_root_funcs(config))
    for layer_ir in ir.modules:
        namespace[layer_ir.name] = decoder_layer_cls

    root_cls = runtime_module(ir)(type("DeepseekV4ForCausalLMRT", (), namespace))
    return root_cls(ir=ir)


__all__ = ["build_runtime_causal_lm"]
