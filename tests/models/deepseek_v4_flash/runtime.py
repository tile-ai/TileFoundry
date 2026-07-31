"""Runtime twin of the DeepSeek-V4-Flash causal-LM tree: hand-written torch
(or, for ``residual_add``, CUDA) kernels for every ``@func`` leaf. See
``build_runtime_causal_lm``.
"""
from __future__ import annotations

import torch

from tests.models.deepseek_v4_flash.model import (
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


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    wf = weight.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(ms + eps) * wf
    return out.to(x.dtype)


def _gather_axis0(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    idx = indices.reshape(-1).long()
    out = torch.index_select(x, 0, idx)
    return out.reshape(tuple(indices.shape) + tuple(x.shape[1:]))


def _block_dequant(
    weight_fp8: torch.Tensor, scale: torch.Tensor, quant_block: int, out_shape: tuple[int, int],
) -> torch.Tensor:
    rows, cols = out_shape
    row_blocks, col_blocks = rows // quant_block, cols // quant_block
    blocks = weight_fp8.to(_BF16).reshape(row_blocks, quant_block, col_blocks, quant_block)
    block_scale = scale.to(_BF16).reshape(row_blocks, 1, col_blocks, 1)
    return (blocks * block_scale).reshape(rows, cols)


# ───────────────────────── CUDA kernel: residual_add ───────────────────────

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
        from torch.utils.cpp_extension import load_inline  # noqa: PLC0415

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
    o_groups = config.o_groups
    o_lora_rank = config.o_lora_rank
    wo_a_in = config.wo_a_in
    wo_a_out = config.wo_a_out
    q_proj = config.q_proj

    @runtime_module(sem)
    class DeepseekV4AttentionRT:
        @runtime_func
        def mla_kv_update(self, hidden, gamma_kv, w_kv, cos_pos, sin_pos):
            kv = torch.matmul(hidden, w_kv)
            kv_n = _rms_norm(kv, gamma_kv)
            kv_4d = kv_n.reshape(1, 1, 1, head_dim)
            kv_nope = kv_4d[..., :nope_dim]
            kv_rope_in = kv_4d[..., nope_dim:]

            # fp8 fake-quant: absmax -> pow2 scale -> clamp -> fp8 round-trip -> dequant, in f32.
            kv_nope_f32 = kv_nope.float()
            kv_nope_blk = kv_nope_f32.reshape(1, 1, 1, kv_quant_blocks, KV_QUANT_BLOCK)
            kv_amax = kv_nope_blk.abs().amax(dim=-1, keepdim=True).clamp_min(FP8E4M3_QUANT_EPS)
            kv_scale = torch.exp2(torch.ceil(torch.log2(kv_amax / FP8E4M3_MAX)))
            kv_scaled = (kv_nope_blk / kv_scale).clamp(-FP8E4M3_MAX, FP8E4M3_MAX)
            kv_q_fp8 = kv_scaled.to(_FP8E4M3)
            kv_dq = kv_q_fp8.to(_F32) * kv_scale
            kv_nope_q = kv_dq.reshape(1, 1, 1, nope_dim).to(_BF16)

            # interleaved-pairs RoPE: rotate in f32, round to bf16 once at the end.
            kv_r0, kv_r1 = kv_rope_in[..., 0::2], kv_rope_in[..., 1::2]
            kv_r0f, kv_r1f = kv_r0.float(), kv_r1.float()
            kv_o0 = (kv_r0f * cos_pos - kv_r1f * sin_pos).to(_BF16)
            kv_o1 = (kv_r0f * sin_pos + kv_r1f * cos_pos).to(_BF16)
            kv_rope_out = torch.stack((kv_o0, kv_o1), dim=-1).reshape(1, 1, 1, rope_dim)
            return torch.cat((kv_nope_q, kv_rope_out), dim=-1)

        @runtime_func
        def mla_attend(
            self, hidden, gamma_q_lora, w_q_a, w_q_b, ones_head_dim, cos_pos, sin_pos,
            kv_cache, kv_new, attn_sink, scale, w_o_a, w_o_b,
        ):
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

            # The cache and the new token as one score row: a decode step
            # attends every position it was given, so no mask. The sink is one
            # more logit in the denominator, dropped before the P@V matmul.
            kv_all = torch.cat((kv_cache, kv_new), dim=1)
            k_h = torch.repeat_interleave(kv_all, n_heads, dim=2).permute(0, 2, 1, 3)
            q_s = (q_final * scale).permute(0, 2, 1, 3)
            scores = torch.matmul(q_s.float(), k_h.float().transpose(2, 3))
            sink = attn_sink.permute(0, 2, 1, 3).expand(1, n_heads, 1, 1)
            probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
            ctx = torch.matmul(probs, k_h.float()).to(_BF16).permute(0, 2, 1, 3)

            ctx_nope = ctx[..., :nope_dim]
            ctx_rope_in = ctx[..., nope_dim:]
            ctx_r0, ctx_r1 = ctx_rope_in[..., 0::2], ctx_rope_in[..., 1::2]
            ctx_r0f, ctx_r1f = ctx_r0.float(), ctx_r1.float()
            ctx_o0 = (ctx_r0f * cos_pos + ctx_r1f * sin_pos).to(_BF16)
            ctx_o1 = (ctx_r1f * cos_pos - ctx_r0f * sin_pos).to(_BF16)
            ctx_rope_out = torch.stack((ctx_o0, ctx_o1), dim=-1).reshape(1, 1, n_heads, rope_dim)
            ctx_final = torch.cat((ctx_nope, ctx_rope_out), dim=-1)
            o_flat = ctx_final.reshape(1, 1, q_proj)

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
            # routing-gate matmul upcasts to f32 here only, matching the semantic body.
            gate = torch.matmul(xt.float(), gate_weight.float().t())
            softplus = torch.log(torch.exp(gate) + 1.0)
            scores = softplus * torch.rsqrt(softplus)
            eids = _gather_axis0(tid2eid, token_ids)
            gweights = torch.gather(scores, 1, eids)
            weight_sum = gweights.sum(dim=-1, keepdim=True)
            gweights = (gweights / weight_sum) * route_scale
            return self.moe_experts_core(x, gweights, eids)

        @runtime_func
        def shared_expert(
            self, x, shared_w1_weight, shared_w1_scale, shared_w3_weight, shared_w3_scale,
            shared_w2_weight, shared_w2_scale,
        ):
            xt = x.reshape(1, dim)
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
        logits = torch.matmul(hidden.reshape(1, dim), lm_head_weight)
        return logits.reshape(1, 1, vocab)

    return {"embed": embed, "final_rms_norm": final_rms_norm, "lm_head": lm_head}


def build_runtime_causal_lm(config: DSV4Config, ir: Module) -> RuntimeModule:
    """The runtime twin of the causal-LM tree at *config*; `ir` is that semantic
    root (its children/entry/functions drive the twin)."""
    if not ir.modules:
        raise ValueError("build_runtime_causal_lm: ir has no decoder layers")
    layer0_ir = ir.modules[0]
    attention_ir = layer0_ir.attention
    moe_ir = layer0_ir.moe

    attention_cls = _build_attention_rt(config, attention_ir)
    moe_cls = _build_moe_rt(config, moe_ir)
    decoder_layer_cls = _build_decoder_layer_rt(layer0_ir, attention_cls, moe_cls)

    namespace: dict[str, object] = dict(_build_root_funcs(config))
    for layer_ir in ir.modules:
        namespace[layer_ir.name] = decoder_layer_cls

    root_cls = runtime_module(ir)(type("DeepseekV4ForCausalLMRT", (), namespace))
    return root_cls(ir=ir)


__all__ = ["build_runtime_causal_lm"]
