"""DeepSeek-V4-Flash causal-LM root: embed, the injected ``decoder_layers``,
final norm and lm_head, plus the decode hooks a generation loop calls."""
from __future__ import annotations

from tests.models.deepseek_v4_flash.config import HF_CONFIG
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf

_MAIN_ROPE: "tuple | None" = None


def _main_rope() -> tuple:
    """HF's own inverse frequencies and attention scaling for the ``main`` rope
    label, computed by ``DeepseekV4RotaryEmbedding`` rather than reproduced.

    A ``sliding_attention`` layer takes ``main`` (plain rope); only the
    compressed layer types take the yarn-scaled ``compress`` label.
    """
    global _MAIN_ROPE
    if _MAIN_ROPE is None:
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
            DeepseekV4RotaryEmbedding,
        )

        _MAIN_ROPE = DeepseekV4RotaryEmbedding.compute_default_rope_parameters(
            HF_CONFIG, layer_type="main"
        )
    return _MAIN_ROPE


def _rope_cos_sin(position: int, *, device):
    """cos / sin for one absolute sequence *position*, each
    ``(config.rope_half,)`` f32, one value per rotated pair."""
    import torch  # noqa: PLC0415

    inv_freq, attention_scaling = _main_rope()
    angles = position * inv_freq.to(torch.float64)
    cos = (angles.cos() * attention_scaling).to(dtype=torch.float32, device=device)
    sin = (angles.sin() * attention_scaling).to(dtype=torch.float32, device=device)
    return cos, sin


def _decode_attn_mask(step: int, *, device):
    """Additive mask ``(1, 1, 1, config.window)`` bf16 for decode step *step*;
    masks not-yet-written slots while the window is filling, none once it has
    wrapped (order doesn't matter to softmax)."""
    import torch  # noqa: PLC0415

    mask = torch.zeros(config.window, dtype=torch.bfloat16, device=device)
    if step < config.window - 1:
        mask[step + 1 :] = float("-inf")
    return mask.view(1, 1, 1, config.window)


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

    layers = decoder_layers

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
        cos, sin = _rope_cos_sin(step, device=device)
        cos_pos = cos.view(1, 1, 1, config.rope_half)
        sin_pos = sin.view(1, 1, 1, config.rope_half)
        attn_mask = _decode_attn_mask(step, device=device)
        scale = torch.full(
            (1, 1, 1, 1), config.head_dim ** -0.5, device=device, dtype=torch.bfloat16,
        )
        ones_head_dim = torch.ones(config.head_dim, device=device, dtype=torch.bfloat16)
        return (
            token_ids, cos_pos, sin_pos, cur_pos, s, past_key_values, attn_mask, scale,
            ones_head_dim,
        )
