"""DeepSeek-V4-Flash causal-LM model tree: embed, per-layer attention/moe
decoder layers, final norm, lm_head, plus the generate()-facing
forward / init_caches / prepare_inputs_for_generation hooks."""
from __future__ import annotations

import json
import math
from pathlib import Path

from tests.models.deepseek_v4_flash.attention import build_attention
from tests.models.deepseek_v4_flash.config import DSV4Config
from tests.models.deepseek_v4_flash.moe import build_moe
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.core.module import Module


def _load_rope_params() -> dict:
    """``rope_theta`` / ``rope_scaling`` (YaRN): not in ``DSV4Config``, read
    directly from ``config.json``."""
    with open(Path(__file__).with_name("config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    scaling = cfg["rope_scaling"]
    if scaling["type"] != "yarn":
        raise ValueError(f"unsupported rope_scaling type {scaling['type']!r}")
    return {
        "theta": float(cfg["rope_theta"]),
        "factor": float(scaling["factor"]),
        "beta_fast": float(scaling["beta_fast"]),
        "beta_slow": float(scaling["beta_slow"]),
        "orig_max_pos": int(scaling["original_max_position_embeddings"]),
    }


_ROPE_PARAMS = _load_rope_params()


def _yarn_inv_freq_and_scale(rope_dim: int):
    """YaRN inverse frequencies (``rope_dim // 2`` of them) and the mscale
    attention scale folded into cos/sin at construction."""
    import torch  # noqa: PLC0415

    theta = _ROPE_PARAMS["theta"]
    factor = _ROPE_PARAMS["factor"]
    beta_fast = _ROPE_PARAMS["beta_fast"]
    beta_slow = _ROPE_PARAMS["beta_slow"]
    orig_max_pos = _ROPE_PARAMS["orig_max_pos"]

    pos_freqs = theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    def _correction_dim(num_rotations: float) -> float:
        return (
            rope_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(theta))

    low = max(math.floor(_correction_dim(beta_fast)), 0)
    high = min(math.ceil(_correction_dim(beta_slow)), rope_dim - 1)
    if low == high:
        high += 0.001  # avoid divide-by-zero below when low == high
    ramp = (torch.arange(rope_dim // 2, dtype=torch.float64) - low) / (high - low)
    extrapolation_factor = 1.0 - ramp.clamp(0, 1)
    inv_freq = (
        inv_freq_interpolation * (1 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return inv_freq, attention_factor


def _rope_cos_sin(config: DSV4Config, position: int, *, device):
    """cos / sin for one absolute sequence *position*, each
    ``(config.rope_half,)`` f32, one value per rotated pair."""
    import torch  # noqa: PLC0415

    inv_freq, attention_factor = _yarn_inv_freq_and_scale(config.rope_dim)
    angles = position * inv_freq
    cos = (angles.cos() * attention_factor).to(dtype=torch.float32, device=device)
    sin = (angles.sin() * attention_factor).to(dtype=torch.float32, device=device)
    return cos, sin


def _decode_attn_mask(config: DSV4Config, step: int, *, device):
    """Additive mask ``(1, 1, 1, config.window)`` bf16 for decode step *step*;
    masks not-yet-written slots while the window is filling, none once it has
    wrapped (order doesn't matter to softmax)."""
    import torch  # noqa: PLC0415

    mask = torch.zeros(config.window, dtype=torch.bfloat16, device=device)
    if step < config.window - 1:
        mask[step + 1 :] = float("-inf")
    return mask.view(1, 1, 1, config.window)


def build_decoder_layer(config: DSV4Config) -> Module:
    """One decoder layer: two pre-norms + residual add, nesting
    ``build_attention(config)`` / ``build_moe(config)`` as its
    ``attention`` / ``moe`` attributes."""

    @module(entry="residual_add")
    class DeepseekV4DecoderLayer:
        @func
        def pre_attn_rms_norm(
            x: Tensor[(1, 1, config.dim), "bf16"],
            pre_attn_norm_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.rms_norm(x, pre_attn_norm_weight)

        @func
        def pre_moe_rms_norm(
            x: Tensor[(1, 1, config.dim), "bf16"],
            pre_moe_norm_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            # ffn_norm.weight is a layer-level tensor (real checkpoint), not part of moe.
            return tf.rms_norm(x, pre_moe_norm_weight)

        @func
        def residual_add(
            a: Tensor[(1, 1, config.dim), "bf16"],
            b: Tensor[(1, 1, config.dim), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.add(a, b)

        attention = build_attention(config)
        moe = build_moe(config)

        def forward(
            self, hidden, cos_pos, sin_pos, cur_pos, s, kv_cache0, attn_mask, scale,
            ones_head_dim, token_ids,
        ):
            attn_in = self.pre_attn_rms_norm(hidden)
            attn_out, kv_cache1 = self.attention(
                attn_in, cos_pos, sin_pos, cur_pos, s, kv_cache0, attn_mask, scale, ones_head_dim,
            )
            h1 = self.residual_add(hidden, attn_out)
            moe_in = self.pre_moe_rms_norm(h1)
            moe_out = self.moe(moe_in, token_ids)
            out = self.residual_add(h1, moe_out)
            return out, kv_cache1

    return DeepseekV4DecoderLayer


def build_causal_lm(config: DSV4Config) -> Module:
    """Full model tree: ``embed`` -> ``config.n_layers`` decoder layers ->
    ``final_rms_norm`` -> ``lm_head``, plus the ``generate()``-facing
    ``forward`` / ``init_caches`` / ``prepare_inputs_for_generation`` hooks."""

    @module(entry="lm_head")
    class DeepseekV4ForCausalLM:
        @func
        def embed(
            table: ConstTensor[(config.vocab, config.dim), "bf16"],
            token_ids: Tensor[(1,), "i64"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.reshape(tf.gather(table, token_ids, axis=0), new_shape=(1, 1, config.dim))

        @func
        def final_rms_norm(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            final_norm_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.rms_norm(hidden, final_norm_weight)

        @func
        def lm_head(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            lm_head_weight: ConstTensor[(config.dim, config.vocab), "bf16"],
        ) -> Tensor[(1, 1, config.vocab), "bf16"]:
            logits = tf.matmul(tf.reshape(hidden, new_shape=(1, config.dim)), lm_head_weight)
            return tf.reshape(logits, new_shape=(1, 1, config.vocab))

        @lm_head.converter("lm_head_weight")
        def _(
            head_weight_raw: ConstTensor[(config.vocab, config.dim), "bf16"],
        ) -> Tensor[(config.dim, config.vocab), "bf16"]:
            # head.weight is (vocab, dim); transpose to match lm_head's (dim, vocab) matmul.
            return tf.transpose(head_weight_raw, perm=(1, 0))

        # Built fresh per index: a shared instance's attention/moe would let one layer's .load() clobber another's.
        layers = tuple(
            build_decoder_layer(config).renamed(f"layer{i}") for i in range(config.n_layers)
        )

        def forward(
            self, token_ids, cos_pos, sin_pos, cur_pos, s, past_key_values, attn_mask, scale,
            ones_head_dim,
        ):
            hidden = self.embed(token_ids)
            new_caches = []
            for i in range(config.n_layers):
                layer = getattr(self, f"layer{i}")
                hidden, new_cache = layer(
                    hidden, cos_pos, sin_pos, cur_pos, s, past_key_values[i], attn_mask, scale,
                    ones_head_dim, token_ids,
                )
                new_caches.append(new_cache)
            normed = self.final_rms_norm(hidden)
            logits = self.lm_head(normed)
            return logits, tuple(new_caches)

        def init_caches(self, device="cuda", mesh=None):
            import torch  # noqa: PLC0415

            return tuple(
                torch.zeros(1, config.window, 1, config.head_dim, dtype=torch.bfloat16, device=device)
                for _ in range(config.n_layers)
            )

        def prepare_inputs_for_generation(self, input_ids, step, past_key_values, device="cuda"):
            import torch  # noqa: PLC0415

            ids = input_ids.reshape(-1)
            token_ids = ids[step].reshape(1).to(device=device, dtype=torch.int64)
            cur_pos = torch.tensor([step % config.window], device=device, dtype=torch.int32)
            s = torch.tensor([1], device=device, dtype=torch.int32)
            cos, sin = _rope_cos_sin(config, step, device=device)
            cos_pos = cos.view(1, 1, 1, config.rope_half)
            sin_pos = sin.view(1, 1, 1, config.rope_half)
            attn_mask = _decode_attn_mask(config, step, device=device)
            scale = torch.full(
                (1, 1, 1, 1), config.head_dim ** -0.5, device=device, dtype=torch.bfloat16,
            )
            ones_head_dim = torch.ones(config.head_dim, device=device, dtype=torch.bfloat16)
            return (
                token_ids, cos_pos, sin_pos, cur_pos, s, past_key_values, attn_mask, scale,
                ones_head_dim,
            )

    return DeepseekV4ForCausalLM


__all__ = ["build_causal_lm", "build_decoder_layer"]
