"""DeepSeek V4 flash decode step: embed -> 1 decoder layer (attention + MoE)
-> final RMSNorm -> lm_head.

Single batch, one new token per step, real transformer **layer 0** structure
throughout (config.json ``compress_ratios[0] == 0``, pure sliding-window
MLA — the only real layer whose attention this fixture's ``attention.py``
structurally implements, see its module docstring): attention is
``attn.mla_kv_update_v2`` / ``attn.mla_attend_v2`` (real dims, a FIXED
``(1, REAL_WINDOW, 1, REAL_HEAD_DIM)`` sliding-window KV cache — no
``DimVar``, unlike the retired GQA placeholder this replaces, since a real
sliding window is a genuinely static-shape cache); MoE is ``moe.py``'s
hash-router variant (``moe_hash_gather`` / ``deepseek_v4_flash_moe_hash`` —
real layers 0..2 per config.json's ``num_hash_layers``), with a
learned-router (``moe_topk`` / ``deepseek_v4_flash_moe``, real layers >= 3)
variant kept selectable via ``run_decode_step``'s ``moe_kind`` for
router-mechanism coverage (``test_decode_step_e2e.py``'s layer-3 test).

The GQA placeholder (``attn.attn_kv_update`` / ``attn.attn_scores``) this
decode step used before is retired from this file and from
``test_decode_step_e2e.py`` — it still lives, untouched, in ``attention.py``
itself (not deleted; see that file's own module docstring for why it stays).

The attention math is two ``@func``s chained in Python (``mla_kv_update_v2``
then ``mla_attend_v2`` — see ``attention.py`` for why a single composed
``@func`` is not used), so the decode step as a whole is a small Python-level
pipeline rather than one composed HIR Function — mirroring
``tests/models/qwen3_5_30b_a3b/test_attention.py``'s ``kv_update`` ->
``scores`` chaining, extended with the embed / MoE / norm / lm_head stages.
Each stage is its own ``evaluate()`` call, so each is independently a leaf
(M1) the caller may or may not have registered a torch-cuda implementation
for; ``deepseek_v4_flash_moe`` / ``deepseek_v4_flash_moe_hash`` additionally
have their own nested leaves (``shared_expert`` etc.) intercepted from
*inside* that one call, per M1's evaluator interception.

RoPE: ``cos``/``sin`` at the new token's absolute position are computed via
``hf_attention_ref.precompute_freqs_cis`` (the official YaRN-scaled RoPE
frequency table; layer 0's ``compress_ratio == 0`` means the YaRN branch is
inactive — see ``rope_freqs_at``) rather than re-deriving that formula here.

Quantization: both the shared expert and the 256 routed experts are real
fp8e4m3 weights with a 128x128-block f32 scale (matching a real
DeepSeek-V4-Flash-FP8 checkpoint — see ``hf_weights.py`` for the on-disk
format this mirrors). ``shared_expert``'s weights are dequanted once via
this module's ``shared_expert`` Module's ``post_init`` (M1), run once and
cached; the downstream reuse of ``moe.py``'s own
``shared_fp8_dequant_w1/w2`` (which still multiply by a "scale") sees an
already-dequantized weight and a neutral (all-ones) scale, so it is an
untouched, harmless no-op on top. Routed expert weights have no
``post_init`` — ``moe.py``'s ``moe_experts_core`` dequants its gathered
(``N_ACT`` of 256) experts' weight + scale inline, every decode step (the
real per-block scale, not a neutral one, so this is a genuine dequant on
the routed path too, not just the shared expert's).
"""
from __future__ import annotations

from typing import Any

import torch

from tests.models.deepseek_v4_flash import attention as attn
from tests.models.deepseek_v4_flash import hf_attention_ref
from tests.models.deepseek_v4_flash.moe import (
    DIM,
    VOCAB,
    combine_expert_outputs,
    deepseek_v4_flash_moe,
    deepseek_v4_flash_moe_hash,
    moe_experts_core,
    moe_hash_gather,
    moe_topk,
    pre_moe_rms_norm,
    shared_expert,
    shared_fp8_dequant_w1,
    shared_fp8_dequant_w2,
)
from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.leaf import ImplementationPackage
from tilefoundry.ir.core.module import Module


@func
def embed(
    table: ConstTensor[(VOCAB, DIM), "bf16"],
    token_ids: Tensor[(1,), "i64"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.reshape(tf.gather(table, token_ids, axis=0), new_shape=(1, 1, DIM))


@func
def residual_add(
    a: Tensor[(1, 1, DIM), "bf16"],
    b: Tensor[(1, 1, DIM), "bf16"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.add(a, b)


@func
def final_rms_norm(
    hidden: Tensor[(1, 1, DIM), "bf16"],
    weight: ConstTensor[(DIM,), "bf16"],
) -> Tensor[(1, 1, DIM), "bf16"]:
    return tf.rms_norm(hidden, weight)


@func
def lm_head(
    hidden: Tensor[(1, 1, DIM), "bf16"],
    weight: ConstTensor[(DIM, VOCAB), "bf16"],
) -> Tensor[(1, 1, VOCAB), "bf16"]:
    logits = tf.matmul(tf.reshape(hidden, new_shape=(1, DIM)), weight)
    return tf.reshape(logits, new_shape=(1, 1, VOCAB))


def _dequant_block_scaled(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Real fp8e4m3 (+ 128x128-block f32 scale) -> bf16 upcast + multiply —
    the same block convention as ``moe.py``'s ``shared_fp8_dequant_w1/w2``."""
    w = weight.to(torch.bfloat16)
    s = scale.to(torch.bfloat16)
    rows, cols = w.shape
    blocks = w.reshape(rows // block, block, cols // block, block)
    block_scale = s.reshape(rows // block, 1, cols // block, 1)
    return (blocks * block_scale).reshape(rows, cols)


def shared_expert_post_init(weights: dict[str, Any]) -> dict[str, Any]:
    """``shared_expert``'s ``post_init`` (M1): real fp8e4m3 + 128x128-block
    f32 scale -> bf16 dequant, run once and cached by ``WeightLoader`` rather
    than on every decode step. The scale keys are replaced by neutral (all-ones, in bf16)
    tensors of the same shape so ``moe.py``'s own ``shared_fp8_dequant_w1/w2``
    (still invoked verbatim on the pure-evaluator path) re-multiplies by 1 —
    an untouched, harmless no-op on top of the already-dequantized weight."""
    out = dict(weights)
    for w_key, s_key in (("w1_weight", "w1_scale"), ("w3_weight", "w3_scale"), ("w2_weight", "w2_scale")):
        out[w_key] = _dequant_block_scaled(weights[w_key], weights[s_key])
        out[s_key] = torch.ones_like(weights[s_key], dtype=torch.bfloat16)
    return out


# ── RoPE (layer 0/1: compress_ratio == 0 -> the non-YaRN rope_theta path) ──
# hf_attention_ref.AttentionRef.__init__: ``if self.compress_ratio: ...
# else: original_seq_len, rope_theta = 0, cfg.rope_theta`` — layer 0 always
# takes the else branch, so YaRN's correction-range smoothing (gated on
# ``original_seq_len > 0``) never runs; REAL_ROPE_FACTOR/BETA_FAST/BETA_SLOW
# below are dead inputs to precompute_freqs_cis in that branch, kept at their
# real config.json values anyway for clarity/robustness rather than
# arbitrary placeholders.
ROPE_TABLE_MAX_POS = 4096  # generous bound; cur_pos must stay < this
REAL_ROPE_THETA = 10000.0  # config.json rope_theta
REAL_ROPE_FACTOR = 16.0  # config.json rope_scaling.factor (unused, see above)
REAL_BETA_FAST = 32  # config.json rope_scaling.beta_fast (unused, see above)
REAL_BETA_SLOW = 1  # config.json rope_scaling.beta_slow (unused, see above)


def rope_freqs_at(cur_pos: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin at absolute position ``cur_pos``, real layer-0/1 RoPE table.
    Borrows ``hf_attention_ref.precompute_freqs_cis`` (``lru_cache``'d there,
    so repeated calls at a fixed table size are cheap) rather than
    re-deriving the YaRN frequency formula here.

    Returned at f32 (``attention.py``'s ``mla_kv_update_v2`` / ``mla_attend_v2``
    signatures both declare ``cos_pos``/``sin_pos`` as f32 -- see those
    functions' rope blocks): ``freqs_cis`` is already complex64 internally
    (built from f32 ``torch.polar`` inputs), so ``.real``/``.imag`` are f32
    natively -- this is a no-op cast, not a widening of stored precision.
    """
    freqs_cis = hf_attention_ref.precompute_freqs_cis(
        attn.REAL_ROPE_DIM, ROPE_TABLE_MAX_POS + 1, 0,
        REAL_ROPE_THETA, REAL_ROPE_FACTOR, REAL_BETA_FAST, REAL_BETA_SLOW,
    )
    row = freqs_cis[cur_pos].to(device)
    cos_pos = row.real.reshape(1, 1, 1, -1).to(torch.float32).contiguous()
    sin_pos = row.imag.reshape(1, 1, 1, -1).to(torch.float32).contiguous()
    return cos_pos, sin_pos


def kv_update_step(
    weights: dict[str, Any],
    hidden: torch.Tensor,
    cur_pos: int,
    kv_cache_prev: torch.Tensor,
    *,
    device: str,
    leaves: dict[str, ImplementationPackage] | None = None,
) -> torch.Tensor:
    """One sliding-window KV-cache update (``attn.mla_kv_update_v2``) at
    absolute position ``cur_pos``: computes that position's RoPE cos/sin and
    the ring-buffer write slot (``cur_pos % REAL_WINDOW`` — ``cache_update``
    itself is a plain linear write with no wraparound, so the modulo happens
    here, mirroring official ``Attention.forward``'s ``start_pos % win``).

    Exposed standalone (not just inlined into ``run_decode_step``) so a
    caller can also use it to fill a sliding-window cache token-by-token from
    empty — see ``test_decode_step_e2e.py``'s KV-cache pre-state helper,
    which does exactly that rather than fabricating a random "pre-existing"
    cache for the real-weight tests.
    """
    cos_pos, sin_pos = rope_freqs_at(cur_pos, device)
    cur_pos_t = torch.tensor([cur_pos % attn.REAL_WINDOW], dtype=torch.int32, device=device)
    s_one = torch.tensor([1], dtype=torch.int32, device=device)
    return evaluate(
        attn.mla_kv_update_v2, hidden,
        weights["layer0.attention.gamma_kv"], weights["layer0.attention.w_kv"],
        cos_pos, sin_pos, kv_cache_prev, cur_pos_t, s_one,
        device=device, leaves=leaves,
    )


# ── Module tree (M0) ─────────────────────────────────────────────────────
# root (unnamed weight prefix) -> layer0 -> {attention, moe -> shared_expert}.
# Weight dict keys are module-path-prefixed (M1), e.g.
# "layer0.moe.shared_expert.w1_weight"; embed / final_rms_norm / lm_head hang
# directly off the root and so are unprefixed ("embed.table", ...).
#
# Two MoE router variants (hash vs learned) share the same "layer0.moe"
# path prefix and the same nested shared_expert_module (Module is a frozen
# dataclass — a child Module object is safe to reference from two different
# parent trees, see its own docstring: "child modules are already fully
# constructed ... sealing does not recurse"); each tree gets its own
# WeightLoader instance in practice, so their post_init caches never mix.
# decode_step_module (hash) is the primary/default real-layer-0 structure;
# decode_step_module_learned exists only to retain learned-router (moe_topk)
# coverage against a real layer >= 3 checkpoint (see test_decode_step_e2e.py).

shared_expert_module = Module(
    name="shared_expert",
    functions=(shared_fp8_dequant_w1, shared_fp8_dequant_w2, shared_expert),
    entry="shared_expert",
    post_init=shared_expert_post_init,
)

moe_module = Module(
    name="moe",
    functions=(pre_moe_rms_norm, moe_experts_core, moe_topk, combine_expert_outputs, deepseek_v4_flash_moe),
    modules=(shared_expert_module,),
    entry="deepseek_v4_flash_moe",
)

moe_hash_module = Module(
    name="moe",
    functions=(pre_moe_rms_norm, moe_experts_core, moe_hash_gather, combine_expert_outputs, deepseek_v4_flash_moe_hash),
    modules=(shared_expert_module,),
    entry="deepseek_v4_flash_moe_hash",
)

attention_module = Module(
    name="attention",
    functions=(attn.mla_kv_update_v2, attn.mla_attend_v2),
    entry="mla_attend_v2",
)

layer0_module = Module(
    name="layer0",
    functions=(residual_add,),
    modules=(attention_module, moe_hash_module),
    entry="residual_add",
)

layer0_module_learned = Module(
    name="layer0",
    functions=(residual_add,),
    modules=(attention_module, moe_module),
    entry="residual_add",
)

decode_step_module = Module(
    name="DeepSeekV4FlashDecodeStep",
    functions=(embed, final_rms_norm, lm_head),
    modules=(layer0_module,),
    entry="lm_head",
)

decode_step_module_learned = Module(
    name="DeepSeekV4FlashDecodeStep",
    functions=(embed, final_rms_norm, lm_head),
    modules=(layer0_module_learned,),
    entry="lm_head",
)


def run_decode_step(
    weights: dict[str, Any],
    *,
    token_ids: torch.Tensor,
    kv_cache_prev: torch.Tensor,
    cur_pos: int,
    device: str,
    moe_kind: str = "hash",
    leaves: dict[str, ImplementationPackage] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one decode step against an already-``WeightLoader.load``-ed
    ``weights`` dict (module-path-prefixed, per M1).

    ``moe_kind`` selects the router HIR graph: ``"hash"`` (real layers 0..2,
    ``moe_hash_gather`` — the default/primary real structure this fixture
    targets) or ``"learned"`` (real layers >= 3, ``moe_topk`` — kept for
    router-mechanism coverage only, see ``test_decode_step_e2e.py``).

    ``kv_cache_prev`` is the sliding-window KV cache (shape
    ``(1, REAL_WINDOW, 1, REAL_HEAD_DIM)``) *before* this step; ``cur_pos``
    is the new token's absolute position (0-indexed) — used both for the
    cache's ring-buffer write slot and for the RoPE angle lookup.

    ``leaves`` is the flat ``{fn_name: ImplementationPackage}`` view
    (``LeafRegistry.by_function_name``); omitted, every stage runs through
    the plain evaluator. Returns ``(logits, kv_cache_new)``.
    """
    if moe_kind not in ("hash", "learned"):
        raise ValueError(f"moe_kind must be 'hash' or 'learned', got {moe_kind!r}")

    hidden = evaluate(embed, weights["embed.table"], token_ids, device=device, leaves=leaves)

    kv_cache_new = kv_update_step(weights, hidden, cur_pos, kv_cache_prev, device=device, leaves=leaves)

    win = attn.REAL_WINDOW
    cos_pos, sin_pos = rope_freqs_at(cur_pos, device)
    # Slots never written yet (window not yet full) are masked out, same as
    # official's get_window_topk_idxs' -1 sentinel for those slots; once
    # cur_pos >= win - 1 the window holds only real (already-written) tokens.
    attn_mask = torch.zeros(1, 1, 1, win, dtype=torch.bfloat16, device=device)
    if cur_pos < win - 1:
        attn_mask[:, :, :, cur_pos + 1:] = float("-inf")
    scale = torch.full((1, 1, 1, 1), attn.REAL_HEAD_DIM ** -0.5, dtype=torch.bfloat16, device=device)
    ones_head_dim = torch.ones(attn.REAL_HEAD_DIM, dtype=torch.bfloat16, device=device)

    attn_out = evaluate(
        attn.mla_attend_v2, hidden,
        weights["layer0.attention.gamma_q_lora"], weights["layer0.attention.w_q_a"],
        weights["layer0.attention.w_q_b"], ones_head_dim, cos_pos, sin_pos,
        kv_cache_new, attn_mask, weights["layer0.attention.attn_sink"], scale,
        weights["layer0.attention.w_o_a"], weights["layer0.attention.w_o_b"],
        device=device, leaves=leaves,
    )
    h1 = evaluate(residual_add, hidden, attn_out, device=device, leaves=leaves)

    if moe_kind == "hash":
        moe_out = evaluate(
            deepseek_v4_flash_moe_hash, h1,
            weights["layer0.moe.rms_weight"], weights["layer0.moe.gate_weight"],
            weights["layer0.moe.tid2eid"], token_ids,
            weights["layer0.moe.routed_w1_weight"], weights["layer0.moe.routed_w1_scale"],
            weights["layer0.moe.routed_w3_weight"], weights["layer0.moe.routed_w3_scale"],
            weights["layer0.moe.routed_w2_weight"], weights["layer0.moe.routed_w2_scale"],
            weights["layer0.moe.shared_expert.w1_weight"], weights["layer0.moe.shared_expert.w1_scale"],
            weights["layer0.moe.shared_expert.w3_weight"], weights["layer0.moe.shared_expert.w3_scale"],
            weights["layer0.moe.shared_expert.w2_weight"], weights["layer0.moe.shared_expert.w2_scale"],
            device=device, leaves=leaves,
        )
    else:
        moe_out = evaluate(
            deepseek_v4_flash_moe, h1,
            weights["layer0.moe.rms_weight"], weights["layer0.moe.gate_weight"],
            weights["layer0.moe.gate_bias"],
            weights["layer0.moe.routed_w1_weight"], weights["layer0.moe.routed_w1_scale"],
            weights["layer0.moe.routed_w3_weight"], weights["layer0.moe.routed_w3_scale"],
            weights["layer0.moe.routed_w2_weight"], weights["layer0.moe.routed_w2_scale"],
            weights["layer0.moe.shared_expert.w1_weight"], weights["layer0.moe.shared_expert.w1_scale"],
            weights["layer0.moe.shared_expert.w3_weight"], weights["layer0.moe.shared_expert.w3_scale"],
            weights["layer0.moe.shared_expert.w2_weight"], weights["layer0.moe.shared_expert.w2_scale"],
            device=device, leaves=leaves,
        )
    h2 = evaluate(residual_add, h1, moe_out, device=device, leaves=leaves)

    normed = evaluate(final_rms_norm, h2, weights["final_rms_norm.weight"], device=device, leaves=leaves)
    logits = evaluate(lm_head, normed, weights["lm_head.weight"], device=device, leaves=leaves)
    return logits, kv_cache_new


__all__ = [
    "ROPE_TABLE_MAX_POS",
    "VOCAB",
    "attention_module",
    "decode_step_module",
    "decode_step_module_learned",
    "embed",
    "final_rms_norm",
    "kv_update_step",
    "layer0_module",
    "layer0_module_learned",
    "lm_head",
    "moe_hash_module",
    "moe_module",
    "residual_add",
    "rope_freqs_at",
    "run_decode_step",
    "shared_expert_module",
    "shared_expert_post_init",
]
