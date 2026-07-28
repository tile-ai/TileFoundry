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


#: How many positions the caller may keep: a query attends ``window``
#: positions counting its own, so the context it is handed is one shorter.
MAX_CTX = config.window - 1

#: The context length the decode loop starts from. A decode step reads a
#: context, so there is one; producing it from a prompt is a prefill, which
#: this package does not state, so it is drawn at a fixed seed instead.
SEED_CTX_LEN = 1
SEED_CTX_SEED = 20260728


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

    def forward(self, token_ids, cos_pos, sin_pos, past_key_values, scale, ones_head_dim):
        """One decode step of the whole model, per-layer cache in and out.

        Each layer hands back its own one-position KV latent; appending it and
        dropping what falls out of the window is the caller's step, and this
        root is the caller. Keeping it here rather than inside a kernel is what
        lets every shape below be expressed in ``ctx_len`` alone.
        """
        import torch  # noqa: PLC0415

        hidden = self.embed(token_ids)
        new_caches = []
        for i in range(config.n_layers):
            layer = getattr(self, f"layer{i}")
            hidden, kv_new = layer(
                hidden, cos_pos, sin_pos, past_key_values[i], scale, ones_head_dim, token_ids,
            )
            grown = torch.cat([past_key_values[i], kv_new], dim=1)
            new_caches.append(grown[:, -MAX_CTX:] if grown.shape[1] > MAX_CTX else grown)
        normed = self.final_rms_norm(hidden)
        logits = self.lm_head(normed)
        return logits, tuple(new_caches)

    def init_caches(self, device="cuda", mesh=None):
        """The per-layer context the decode loop starts from, drawn at a fixed
        seed so every caller of this model starts from the same one."""
        import torch  # noqa: PLC0415

        generator = torch.Generator(device=device).manual_seed(SEED_CTX_SEED)
        return tuple(
            (
                torch.randn(
                    1, SEED_CTX_LEN, 1, config.head_dim,
                    generator=generator, device=device, dtype=torch.float32,
                )
                * 0.1
            ).to(torch.bfloat16)
            for _ in range(config.n_layers)
        )

    def prepare_inputs_for_generation(self, input_ids, step, past_key_values, device="cuda"):
        import torch  # noqa: PLC0415

        ids = input_ids.reshape(-1)
        token_ids = ids[step].reshape(1).to(device=device, dtype=torch.int64)
        # The step's own absolute position: the seed context occupies the ones
        # before it, and rotation is what ties a key to the position it was
        # written at.
        cos, sin = _rope_cos_sin(SEED_CTX_LEN + step, device=device)
        cos_pos = cos.view(1, 1, 1, config.rope_half)
        sin_pos = sin.view(1, 1, 1, config.rope_half)
        scale = torch.full(
            (1, 1, 1, 1), config.head_dim ** -0.5, device=device, dtype=torch.bfloat16,
        )
        ones_head_dim = torch.ones(config.head_dim, device=device, dtype=torch.bfloat16)
        return (token_ids, cos_pos, sin_pos, past_key_values, scale, ones_head_dim)
