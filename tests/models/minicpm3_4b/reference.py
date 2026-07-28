"""The executable semantics one MiniCPM3-4B decode step is held to.

Inputs and oracle are one pair on purpose. The oracle is a Hugging Face layer
with random weights, so a factory that returned only tensors would leave the
reference free to score them against a differently initialised layer. What
`inputs` returns therefore carries both: the arguments the evaluator is called
with, and the module those arguments were drawn from.

Everything is seeded. The same call returns the same weights and the same
activations, so a disagreement is a disagreement about the compiler rather than
about which random draw each side happened to get.

One step is drawn at a stated context length: the context's hidden states are
drawn, the KV cache is built from them, and the token being decoded is the one
that follows. The cache is passed as plain tensors and the oracle is taken from a
full-sequence forward's last position, so neither side constructs a Hugging Face
cache object.

``residual_scale`` is read off the HF layer rather than derived, because
``scale_depth / sqrt(num_hidden_layers)`` differs between the one-layer component
fixture and the real 62-layer stack -- and a reference that recomputed it would
be free to disagree with the layer it is scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tests.models.minicpm3_4b import config

#: Whatever runs this states its own device; the oracle is a CPU f32 baseline.
DEVICE = "cpu"

#: Seeds, named so a change to either is a visible change to the reference.
WEIGHT_SEED = 0
ACTIVATION_SEED = 1

#: The context length a case is drawn at unless it states another. Small enough
#: to keep the oracle's full-sequence forward cheap, and not a multiple of the
#: head count, so an index arithmetic error cannot coincide with a head boundary.
CTX_LEN = 24


@dataclass(frozen=True)
class DecodeStepInputs:
    """One drawn step: the evaluator's arguments and the layer behind them."""

    args: tuple[torch.Tensor, ...]
    layer: object
    ctx_len: int
    hidden_ctx: torch.Tensor
    hidden_new: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor


def decode_step_inputs(*, ctx_len: int = CTX_LEN, device: str = DEVICE) -> DecodeStepInputs:
    """One deterministic decode step over a *ctx_len*-token context."""
    layer = config.build_hf_layer(seed=WEIGHT_SEED, device=device)
    cfg = config.build_hf_config()
    cos_cache, sin_cache = config.rope_caches(cfg, config.REAL.max_pos, device=device)
    scale = torch.full((1, 1, 1, 1), layer.self_attn.scaling, device=device)
    residual_scale = torch.full((1, 1, 1), layer.residual_scale, device=device)

    torch.manual_seed(ACTIVATION_SEED)
    drawn = torch.randn(1, ctx_len + 1, config.REAL.hidden, device=device) * 0.1
    hidden_ctx, hidden_new = drawn[:, :ctx_len], drawn[:, ctx_len:]
    k_cache, v_cache = config.context_kv(layer, hidden_ctx, device=device)

    # The token being decoded sits immediately after the context.
    pos_ids = torch.tensor([ctx_len], device=device, dtype=torch.int32)

    attention, mlp = layer.self_attn, layer.mlp
    return DecodeStepInputs(
        args=(
            hidden_new,
            layer.input_layernorm.weight,
            config.linear_weight(attention.q_a_proj),
            attention.q_a_layernorm.weight,
            config.linear_weight(attention.q_b_proj),
            config.linear_weight(attention.kv_a_proj_with_mqa),
            attention.kv_a_layernorm.weight,
            config.linear_weight(attention.kv_b_proj),
            cos_cache,
            sin_cache,
            pos_ids,
            k_cache,
            v_cache,
            scale,
            config.linear_weight(attention.o_proj),
            layer.post_attention_layernorm.weight,
            config.linear_weight(mlp.gate_proj),
            config.linear_weight(mlp.up_proj),
            config.linear_weight(mlp.down_proj),
            residual_scale,
        ),
        layer=layer,
        ctx_len=ctx_len,
        hidden_ctx=hidden_ctx,
        hidden_new=hidden_new,
        k_cache=k_cache,
        v_cache=v_cache,
    )


def decode_step_oracle(inputs: DecodeStepInputs) -> torch.Tensor:
    """What Hugging Face's own layer produces for the same drawn step."""
    return config.decode_reference(inputs.layer, inputs.hidden_ctx, inputs.hidden_new)


def appended_cache_oracle(inputs: DecodeStepInputs) -> tuple[torch.Tensor, torch.Tensor]:
    """The cache the step's caller should hold afterwards.

    Built the same way the input cache was, over the context with the decoded
    token appended: the kernel's returned key and value are correct exactly when
    appending them reproduces this.
    """
    return config.context_kv(
        inputs.layer,
        torch.cat([inputs.hidden_ctx, inputs.hidden_new], dim=1),
        device=inputs.hidden_ctx.device.type,
    )


@dataclass(frozen=True)
class DecoderStepInputs:
    """One drawn step of the complete decoder, and the stack behind it."""

    model: object
    ctx_len: int
    hidden_ctx: torch.Tensor
    hidden_new: torch.Tensor
    caches: list
    weights: list
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor
    pos_ids: torch.Tensor
    scale: torch.Tensor
    residual_scale: torch.Tensor

    @property
    def args(self) -> tuple:
        """What `run_decoder_step` passes on, for the wiring check to count."""
        return (
            self.hidden_new, self.cos_cache, self.sin_cache, self.pos_ids,
            self.scale, self.residual_scale, self.weights, self.caches,
        )


def decoder_step_inputs(*, ctx_len: int = CTX_LEN, device: str = "cuda"):
    """One decode step of the complete decoder, over a *ctx_len*-token context.

    The whole stack, so the drawn problem is per layer: each layer's weights and
    each layer's own cache, in layer order. Built from a decoder whose layers are
    the production count, because layer order and the residual thread between
    layers are the things this boundary exists to observe -- and because
    ``residual_scale`` is only the production value at the production depth.
    """
    model = config.build_hf_decoder(seed=WEIGHT_SEED, device=device)
    torch.manual_seed(ACTIVATION_SEED)
    drawn = torch.randn(1, ctx_len + 1, config.REAL.hidden, device=device) * 0.1
    hidden_ctx, hidden_new = drawn[:, :ctx_len], drawn[:, ctx_len:]

    cfg = config.build_hf_config()
    cos_cache, sin_cache = config.rope_caches(cfg, config.REAL.max_pos, device=device)
    first = model.layers[0]
    return DecoderStepInputs(
        model=model,
        ctx_len=ctx_len,
        hidden_ctx=hidden_ctx,
        hidden_new=hidden_new,
        caches=config.decoder_context_kv(model, hidden_ctx, device=device),
        weights=[layer_weights(layer) for layer in model.layers],
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        pos_ids=torch.tensor([ctx_len], device=device, dtype=torch.int32),
        scale=torch.full((1, 1, 1, 1), first.self_attn.scaling, device=device),
        residual_scale=torch.full((1, 1, 1), first.residual_scale, device=device),
    )


def layer_weights(layer) -> tuple:
    """One layer's weights, in the order its decode step takes them."""
    attention, mlp = layer.self_attn, layer.mlp
    return (
        layer.input_layernorm.weight,
        config.linear_weight(attention.q_a_proj),
        attention.q_a_layernorm.weight,
        config.linear_weight(attention.q_b_proj),
        config.linear_weight(attention.kv_a_proj_with_mqa),
        attention.kv_a_layernorm.weight,
        config.linear_weight(attention.kv_b_proj),
        config.linear_weight(attention.o_proj),
        layer.post_attention_layernorm.weight,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
    )


def run_decoder_step(inputs: DecoderStepInputs):
    """The complete decoder over *inputs*, through the Evaluator."""
    from tests.models.minicpm3_4b.decoder import build_decoder  # noqa: PLC0415

    decoder = build_decoder().bind_final_norm(inputs.model.norm.weight)
    return decoder.forward(*inputs.args)


def decoder_step_oracle(inputs: DecoderStepInputs) -> torch.Tensor:
    """What Hugging Face's own stack produces for the same drawn step."""
    return config.decoder_decode_reference(
        inputs.model, inputs.hidden_ctx, inputs.hidden_new
    )


__all__ = [
    "ACTIVATION_SEED",
    "CTX_LEN",
    "DEVICE",
    "WEIGHT_SEED",
    "DecodeStepInputs",
    "DecoderStepInputs",
    "appended_cache_oracle",
    "decode_step_inputs",
    "decode_step_oracle",
    "decoder_step_inputs",
    "decoder_step_oracle",
    "layer_weights",
    "run_decoder_step",
]
