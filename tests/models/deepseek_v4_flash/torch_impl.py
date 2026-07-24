"""Torch-cuda leaf implementations (M1's ``ImplementationPackage(language="torch")``
backend) for the DeepSeek V4 flash decode step.

Each function mirrors its HIR counterpart's math 1:1, op for op (matmul ->
``torch.matmul``, ``tf.rms_norm`` -> upcast-to-f32 reduction, ``tf.slice``
with a stride -> plain Python strided indexing, ``tf.softmax`` -> upcast then
downcast, ...; see each HIR op's ``register_eval`` in
``tilefoundry.ir.hir.*`` for the exact per-op semantics this mirrors) — same
positional args as the HIR ``Call`` site (so it drops straight into the
evaluator's Call interception, see ``tilefoundry.evaluator.leaf`` /
``interpreter.py``), same outputs — so pure-evaluator and leaf-registered
execution agree within ``test_decode_step_e2e.py``'s tolerance.

``moe_topk`` / ``moe_hash_gather``'s leaves subsume ``moe_experts_core``
(their only caller): the routed-expert compute never runs through the plain
evaluator's ``Gather`` / ``MatMul`` handlers when the router itself is
intercepted, since interception skips recursing into the callee's body
entirely — so ``moe_experts_core`` has no separate registration.
``deepseek_v4_flash_moe`` / ``deepseek_v4_flash_moe_hash`` themselves (the
top-level composed MoE Functions) are never registered either: they always
run via the plain evaluator, composing their own (possibly leaf-intercepted)
nested calls — that is what exercises M1's "interception at any nesting
depth" property.

Routed experts carry a real fp8e4m3 + 128x128-block f32 scale (matching the
real checkpoint format, see ``hf_weights.py``); ``torch_moe_topk`` /
``torch_moe_hash_gather`` dequant them themselves
(``_dequant_gathered_experts``), same block convention as ``moe.py``'s
``moe_experts_core``. Shared-expert weights are already bf16-valued with a
neutral (ones) scale by the time any of this runs (see
``decode_step.py``'s module docstring and ``shared_expert_post_init``), so
``torch_shared_expert``'s scale positional args are accepted (to match the
HIR Call's arity) but never used.

``build_full_leaf_registry(root)`` takes the module tree to register
against — ``decode_step.decode_step_module`` (hash router, real layer 0..2 —
the default) or ``decode_step.decode_step_module_learned`` (learned router,
real layer >= 3, kept for router-mechanism coverage) — and only registers
the leaves in ``_LEAF_IMPLS`` that ``root`` actually contains (the two trees
have disjoint router leaves: ``moe_hash_gather`` vs ``moe_topk``).
"""
from __future__ import annotations

import torch

from tests.models.deepseek_v4_flash import attention as attn
from tests.models.deepseek_v4_flash.decode_step import decode_step_module
from tests.models.deepseek_v4_flash.moe import DIM, MOE_INTER, N_ACT, ROUTE_SCALE, SWIGLU_LIMIT
from tilefoundry.evaluator.leaf import ImplementationPackage, LeafRegistry, leaf_paths


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(ms + eps) * weight.float()
    return out.to(x.dtype)


def _fake_quant_fp8_block(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mirrors ``hf_attention_ref._fake_quant_fp8_block`` (``round_scale=True``
    only, ported rather than imported -- see this file's module docstring on
    each leaf file owning its own self-contained implementation): per-
    ``block_size`` (last dim) absmax -> power-of-2 scale
    (``exp2(ceil(log2(amax/448)))``) -> divide -> clamp -> real
    ``torch.float8_e4m3fn`` round-trip -> multiply back, returned in ``x``'s
    original dtype. Used by ``torch_mla_kv_update_v2`` to mirror
    ``attention.mla_kv_update_v2``'s HIR fake-quant op for op."""
    orig_dtype = x.dtype
    *lead, n = x.shape
    xf = x.float().reshape(*lead, n // block_size, block_size)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    fp8_max = 448.0
    scale = torch.exp2(torch.ceil(torch.log2(amax / fp8_max)))
    q = (xf / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).to(torch.float32)
    y = (q * scale).reshape(*lead, n)
    return y.to(orig_dtype)


def torch_embed(table: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    dim = table.shape[-1]
    return table.index_select(0, token_ids.long()).reshape(1, 1, dim)


def torch_residual_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def torch_mla_kv_update_v2(
    hidden, gamma_kv, w_kv, cos_pos, sin_pos, kv_cache0, cur_pos, s,
):
    """Mirrors ``attention.mla_kv_update_v2`` op for op (see that function's
    body for the official semantics each step reproduces)."""
    kv = torch.matmul(hidden, w_kv)
    kv_n = _rms_norm(kv, gamma_kv)
    kv_4d = kv_n.reshape(1, 1, 1, attn.REAL_HEAD_DIM)
    kv_nope = kv_4d[..., : attn.REAL_NOPE_DIM]
    kv_nope = _fake_quant_fp8_block(kv_nope, attn.KV_QUANT_BLOCK)
    kv_rope_in = kv_4d[..., attn.REAL_NOPE_DIM :]
    kv_r0 = kv_rope_in[..., 0::2]
    kv_r1 = kv_rope_in[..., 1::2]
    # f32 upcast for the rotation, single rounding back to bf16 -- mirrors
    # attention.mla_kv_update_v2's HIR rope block op for op (cos_pos/sin_pos
    # arrive as f32; kv_r0/kv_r1 explicitly upcast rather than relying on
    # torch's implicit bf16*f32->f32 promotion, so this matches the HIR's
    # explicit tf.cast exactly).
    kv_r0_f32 = kv_r0.float()
    kv_r1_f32 = kv_r1.float()
    kv_o0 = (kv_r0_f32 * cos_pos - kv_r1_f32 * sin_pos).to(torch.bfloat16)
    kv_o1 = (kv_r0_f32 * sin_pos + kv_r1_f32 * cos_pos).to(torch.bfloat16)
    kv_o0 = kv_o0.reshape(1, 1, 1, attn.REAL_ROPE_HALF, 1)
    kv_o1 = kv_o1.reshape(1, 1, 1, attn.REAL_ROPE_HALF, 1)
    kv_interleaved = torch.cat([kv_o0, kv_o1], dim=-1)
    kv_rope_out = kv_interleaved.reshape(1, 1, 1, attn.REAL_ROPE_DIM)
    kv_final = torch.cat([kv_nope, kv_rope_out], dim=-1)

    cur = int(cur_pos.reshape(-1)[0].item())
    slen = int(s.reshape(-1)[0].item())
    out = kv_cache0.clone()
    out[:, cur : cur + slen] = kv_final[:, :slen].to(out.dtype)
    return out


def torch_mla_attend_v2(
    hidden, gamma_q_lora, w_q_a, w_q_b, ones_head_dim, cos_pos, sin_pos,
    kv_cache, attn_mask, attn_sink, scale, w_o_a, w_o_b,
):
    """Mirrors ``attention.mla_attend_v2`` op for op (see that function's
    body for the official semantics each step reproduces)."""
    q_lat = _rms_norm(torch.matmul(hidden, w_q_a), gamma_q_lora)
    q_full = torch.matmul(q_lat, w_q_b)
    q = q_full.reshape(1, 1, attn.REAL_N_HEADS, attn.REAL_HEAD_DIM)
    q_rescaled = _rms_norm(q, ones_head_dim)
    q_nope = q_rescaled[..., : attn.REAL_NOPE_DIM]
    q_rope_in = q_rescaled[..., attn.REAL_NOPE_DIM :]
    q_r0 = q_rope_in[..., 0::2]
    q_r1 = q_rope_in[..., 1::2]
    # f32 upcast for the rotation, single rounding back to bf16 (see
    # torch_mla_kv_update_v2's identical rope block for the rationale).
    q_r0_f32 = q_r0.float()
    q_r1_f32 = q_r1.float()
    q_o0 = (q_r0_f32 * cos_pos - q_r1_f32 * sin_pos).to(torch.bfloat16)
    q_o1 = (q_r0_f32 * sin_pos + q_r1_f32 * cos_pos).to(torch.bfloat16)
    q_o0 = q_o0.reshape(1, 1, attn.REAL_N_HEADS, attn.REAL_ROPE_HALF, 1)
    q_o1 = q_o1.reshape(1, 1, attn.REAL_N_HEADS, attn.REAL_ROPE_HALF, 1)
    q_interleaved = torch.cat([q_o0, q_o1], dim=-1)
    q_rope_out = q_interleaved.reshape(1, 1, attn.REAL_N_HEADS, attn.REAL_ROPE_DIM)
    q_final = torch.cat([q_nope, q_rope_out], dim=-1)

    k_b = torch.repeat_interleave(kv_cache, attn.REAL_N_HEADS, dim=2)
    q_h = q_final.permute(0, 2, 1, 3)
    k_h = k_b.permute(0, 2, 1, 3)
    q_s = q_h * scale
    k_t = k_h.transpose(-1, -2)
    scores = torch.matmul(q_s, k_t) + attn_mask

    scores_ext = torch.cat([scores, attn_sink], dim=-1)
    probs_ext = torch.softmax(scores_ext.float(), dim=-1).to(scores_ext.dtype)
    probs = probs_ext[:, :, :, : attn.REAL_WINDOW]
    ctx = torch.matmul(probs, k_h)

    ctx_nope = ctx[..., : attn.REAL_NOPE_DIM]
    ctx_rope_in = ctx[..., attn.REAL_NOPE_DIM :]
    ctx_r0 = ctx_rope_in[..., 0::2]
    ctx_r1 = ctx_rope_in[..., 1::2]
    # f32 upcast for the rotation, single rounding back to bf16 (see
    # torch_mla_kv_update_v2's identical rope block for the rationale).
    ctx_r0_f32 = ctx_r0.float()
    ctx_r1_f32 = ctx_r1.float()
    ctx_o0 = (ctx_r0_f32 * cos_pos + ctx_r1_f32 * sin_pos).to(torch.bfloat16)
    ctx_o1 = (ctx_r1_f32 * cos_pos - ctx_r0_f32 * sin_pos).to(torch.bfloat16)
    ctx_o0 = ctx_o0.reshape(1, attn.REAL_N_HEADS, 1, attn.REAL_ROPE_HALF, 1)
    ctx_o1 = ctx_o1.reshape(1, attn.REAL_N_HEADS, 1, attn.REAL_ROPE_HALF, 1)
    ctx_interleaved = torch.cat([ctx_o0, ctx_o1], dim=-1)
    ctx_rope_out = ctx_interleaved.reshape(1, attn.REAL_N_HEADS, 1, attn.REAL_ROPE_DIM)
    ctx_final = torch.cat([ctx_nope, ctx_rope_out], dim=-1)

    attn_out_heads_last = ctx_final.permute(0, 2, 1, 3)
    o_flat = attn_out_heads_last.reshape(1, 1, attn.REAL_Q_PROJ)

    outs = []
    for g in range(attn.REAL_O_GROUPS):
        o_g = o_flat[:, :, g * attn.REAL_WO_A_IN : (g + 1) * attn.REAL_WO_A_IN]
        outs.append(torch.matmul(o_g, w_o_a[g]))
    y = torch.cat(outs, dim=-1)
    return torch.matmul(y, w_o_b)


def torch_pre_moe_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _rms_norm(x, weight)


def _dequant_gathered_experts(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """``(1, N_ACT, R, C)`` fp8e4m3 gathered-expert weight x ``(1, N_ACT,
    R//block, C//block)`` f32 block scale -> bf16, same 128x128-block
    convention as ``moe.py``'s ``moe_experts_core`` (real checkpoint format:
    routed experts are no longer a neutral/ones scale — see hf_weights.py)."""
    b, n, rows, cols = weight.shape
    w = weight.to(torch.bfloat16).reshape(b, n, rows // block, block, cols // block, block)
    s = scale.to(torch.bfloat16).reshape(b, n, rows // block, 1, cols // block, 1)
    return (w * s).reshape(b, n, rows, cols)


def _moe_experts_core_torch(xt, gweights, eids, w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale):
    """Shared routed-expert compute tail for ``torch_moe_topk`` /
    ``torch_moe_hash_gather`` (mirrors ``moe.py``'s ``moe_experts_core``) —
    both routers gather the same expert weight format, differing only in how
    ``eids`` / ``gweights`` are derived."""
    w1 = _dequant_gathered_experts(w1_weight[eids], w1_scale[eids])  # (1, N_ACT, MOE_INTER, DIM)
    w3 = _dequant_gathered_experts(w3_weight[eids], w3_scale[eids])
    w2 = _dequant_gathered_experts(w2_weight[eids], w2_scale[eids])  # (1, N_ACT, DIM, MOE_INTER)
    token = xt.to(torch.bfloat16).reshape(1, 1, DIM, 1)
    gate_value = torch.matmul(w1, token).reshape(1, N_ACT, MOE_INTER).float()
    up_value = torch.matmul(w3, token).reshape(1, N_ACT, MOE_INTER).float()
    up_value = torch.clamp(up_value, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    gate_value = torch.clamp(gate_value, max=SWIGLU_LIMIT)
    hidden = (gate_value * torch.sigmoid(gate_value)) * up_value
    hidden = hidden.to(torch.bfloat16).reshape(1, N_ACT, MOE_INTER, 1)
    expert_output = torch.matmul(w2, hidden).reshape(1, N_ACT, DIM).float()
    weighted = expert_output * gweights.reshape(1, N_ACT, 1)
    return weighted.to(torch.bfloat16)


def torch_moe_topk(
    x, gate_weight, gate_bias, w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
):
    xt = x.reshape(1, DIM)
    gate = torch.matmul(xt.float(), gate_weight.float().t())
    softplus = torch.log(torch.exp(gate) + 1.0)
    scores = softplus * torch.rsqrt(softplus)
    selection = scores + gate_bias.reshape(1, -1).float()
    _, eids = torch.topk(selection, k=N_ACT, dim=-1)
    gweights = torch.gather(scores, 1, eids)
    weight_sum = gweights.sum(dim=-1, keepdim=True)
    gweights = (gweights / weight_sum) * ROUTE_SCALE
    return _moe_experts_core_torch(xt, gweights, eids, w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale)


def torch_moe_hash_gather(
    x, gate_weight, tid2eid, token_ids, w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
):
    """Mirrors ``moe.moe_hash_gather``: hash-router selection (per-token-id
    ``tid2eid`` lookup, no bias) instead of ``torch_moe_topk``'s learned
    top-k, sharing the same routed-expert compute tail."""
    xt = x.reshape(1, DIM)
    gate = torch.matmul(xt.float(), gate_weight.float().t())
    softplus = torch.log(torch.exp(gate) + 1.0)
    scores = softplus * torch.rsqrt(softplus)
    eids = tid2eid[token_ids.long()]  # (1, N_ACT) i64
    gweights = torch.gather(scores, 1, eids)
    weight_sum = gweights.sum(dim=-1, keepdim=True)
    gweights = (gweights / weight_sum) * ROUTE_SCALE
    return _moe_experts_core_torch(xt, gweights, eids, w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale)


def torch_shared_expert(x, w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale):
    del w1_scale, w3_scale, w2_scale  # neutral (ones) post-post_init — see module docstring
    xt = x.reshape(1, DIM)
    gate = torch.matmul(xt, w1_weight.t()).float()
    up = torch.matmul(xt, w3_weight.t()).float()
    up = torch.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    hidden = ((gate * torch.sigmoid(gate)) * up).to(torch.bfloat16)
    output = torch.matmul(hidden, w2_weight.t()).to(torch.bfloat16)
    return output.reshape(1, 1, DIM)


def torch_combine_expert_outputs(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    return routed + shared


def torch_final_rms_norm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _rms_norm(hidden, weight)


def torch_lm_head(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dim = hidden.shape[-1]
    vocab = weight.shape[-1]
    logits = torch.matmul(hidden.reshape(1, dim), weight)
    return logits.reshape(1, 1, vocab)


_LEAF_IMPLS = {
    "embed": torch_embed,
    "residual_add": torch_residual_add,
    "mla_kv_update_v2": torch_mla_kv_update_v2,
    "mla_attend_v2": torch_mla_attend_v2,
    "pre_moe_rms_norm": torch_pre_moe_rms_norm,
    "moe_topk": torch_moe_topk,
    "moe_hash_gather": torch_moe_hash_gather,
    "shared_expert": torch_shared_expert,
    "combine_expert_outputs": torch_combine_expert_outputs,
    "final_rms_norm": torch_final_rms_norm,
    "lm_head": torch_lm_head,
}


def build_full_leaf_registry(root=decode_step_module) -> LeafRegistry:
    """Register every leaf in ``_LEAF_IMPLS`` that ``root`` actually contains
    at its module path in ``root`` (M3's "all leaves registered" run). The
    hash-router tree (``decode_step_module``, the default) and the
    learned-router tree (``decode_step_module_learned``) have disjoint router
    leaves (``moe_hash_gather`` vs ``moe_topk``) — a name from ``_LEAF_IMPLS``
    absent from ``root``'s own tree is silently skipped rather than raising,
    so this one function serves both trees."""
    paths = leaf_paths(root)
    registry = LeafRegistry()
    for name, impl_fn in _LEAF_IMPLS.items():
        if name not in paths:
            continue
        registry.register(
            paths[name], name, ImplementationPackage(language="torch", fn_or_source=impl_fn, entry=name),
        )
    return registry


__all__ = ["build_full_leaf_registry"] + [f"torch_{name}" for name in _LEAF_IMPLS]
