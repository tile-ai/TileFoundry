"""One DeepSeek-V4-Flash decoder layer: two pre-norms + residual add, nesting the
injected ``attention_module`` / ``moe_module`` as its ``attention`` / ``moe``
children."""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf


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

    attention = attention_module
    moe = moe_module

    def forward(self, hidden, cos_pos, sin_pos, kv_cache, scale, ones_head_dim, token_ids):
        attn_in = self.pre_attn_rms_norm(hidden)
        attn_out, kv_new = self.attention(
            attn_in, cos_pos, sin_pos, kv_cache, scale, ones_head_dim,
        )
        h1 = self.residual_add(hidden, attn_out)
        moe_in = self.pre_moe_rms_norm(h1)
        moe_out = self.moe(moe_in, token_ids)
        out = self.residual_add(h1, moe_out)
        return out, kv_new
