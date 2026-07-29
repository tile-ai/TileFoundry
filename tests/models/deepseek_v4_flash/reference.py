"""The executable semantics one DeepSeek-V4-Flash decode step is held to.

Inputs and oracle are one pair on purpose. The oracle is a Hugging Face
attention module with random weights, so a factory that returned only tensors
would leave the reference free to score them against a differently initialised
module. What `attention_step_inputs` returns therefore carries both: the
arguments the step is run with, and the module those arguments were drawn from.

Everything is seeded. The same call returns the same weights and the same
activations, so a disagreement is a disagreement about the compiler rather than
about which random draw each side happened to get.

One step is drawn at a stated context length: the context's hidden states are
drawn, the KV cache is built from them through the module's own norm,
projection and rotation, and the token being decoded is the one that follows.
The cache is passed as a plain tensor and the oracle is taken from a
full-sequence forward's last position, so neither side constructs a Hugging Face
cache object.

The boundary is the attention submodule rather than one Function of it, because
a decode step here is two Functions -- the KV latent this token writes, and the
attention over the context it was given -- composed by the module's own
`forward`, with the weights bound by name the way the checkpoint binds them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tests.models.deepseek_v4_flash import config as shape
from tests.models.deepseek_v4_flash.model import DeepseekV4Attention
from tilefoundry.runtime import DictResource

#: The model is bf16 with an fp8 KV cache; the oracle is asked in the dtype the
#: model is authored in, and `test_attention_decode.py` states separately what
#: it costs against an f32 accumulation of the same numbers.
DTYPE = torch.bfloat16
DEVICE = "cuda"

#: Seeds, named so a change to either is a visible change to the reference.
WEIGHT_SEED = 0
ACTIVATION_SEED = 1

#: The context a decode step is drawn over. This is a sliding-window layer, so
#: the longest context it can attend is one shorter than the window -- stated
#: rather than minimised, because a decode kernel's cost is dominated by the
#: cache it streams and a shorter one would report a profile no deployment has.
CTX_LEN = shape.REAL.window - 1


@dataclass(frozen=True)
class DecodeStepInputs:
    """One drawn step: the step's arguments and the module behind them."""

    args: tuple
    weights: dict
    layer: object
    ctx_len: int
    hidden_ctx: torch.Tensor
    hidden_new: torch.Tensor
    kv_cache: torch.Tensor


def _weights_of(layer) -> dict:
    """*layer*'s weights under the names the kernels bind them by.

    The kernel convention is `x[1, S, in] @ w[1, in, out]`, so every transpose
    below is weight preprocessing and belongs on this side of the boundary. The
    grouped output projection is stated the way Hugging Face's own
    `DeepseekV4GroupedLinear.forward` states it, for the same reason.
    """
    real = shape.REAL
    return {
        "gamma_kv": layer.kv_norm.weight.detach(),
        "w_kv": layer.kv_proj.weight.detach().t().contiguous(),
        "gamma_q_lora": layer.q_a_norm.weight.detach(),
        "w_q_a": layer.q_a_proj.weight.detach().t().contiguous(),
        "w_q_b": layer.q_b_proj.weight.detach().t().contiguous(),
        "attn_sink": layer.sinks.detach().reshape(1, 1, real.n_heads, 1).float(),
        "w_o_a": layer.o_a_proj.weight.detach()
        .view(real.o_groups, real.o_lora_rank, real.wo_a_in)
        .transpose(1, 2)
        .contiguous(),
        "w_o_b": layer.o_b_proj.weight.detach().t().contiguous(),
    }


def attention_step_inputs(*, ctx_len: int = CTX_LEN, device: str = DEVICE) -> DecodeStepInputs:
    """One deterministic decode step over a *ctx_len*-token context."""
    real = shape.REAL
    layer = shape.build_hf_attention(seed=WEIGHT_SEED, device=device, dtype=DTYPE)

    torch.manual_seed(ACTIVATION_SEED)
    drawn = (torch.randn(1, ctx_len + 1, real.dim, device=device) * 0.1).to(DTYPE)
    hidden_ctx, hidden_new = drawn[:, :ctx_len], drawn[:, ctx_len:]
    kv_cache = shape.context_kv(layer, hidden_ctx)

    # The token being decoded sits immediately after the context.
    cos, sin = shape.rope_caches(ctx_len + 1, device)
    cos_pos = cos[:, ctx_len:].reshape(1, 1, 1, real.rope_half).float()
    sin_pos = sin[:, ctx_len:].reshape(1, 1, 1, real.rope_half).float()

    return DecodeStepInputs(
        args=(
            hidden_new,
            cos_pos,
            sin_pos,
            kv_cache,
            torch.full((1, 1, 1, 1), real.head_dim**-0.5, device=device, dtype=DTYPE),
            torch.ones(real.head_dim, device=device, dtype=DTYPE),
        ),
        weights=_weights_of(layer),
        layer=layer,
        ctx_len=ctx_len,
        hidden_ctx=hidden_ctx,
        hidden_new=hidden_new,
        kv_cache=kv_cache,
    )


def run_attention_step(inputs: DecodeStepInputs):
    """The attention submodule over *inputs*, through the Evaluator.

    A freshly copied module every call, weights bound by name from the drawn
    step: the description under test is the one the checkpoint pipeline binds
    into, not a second copy that takes its weights positionally.
    """
    loaded = DeepseekV4Attention.cloned().load(DictResource(inputs.weights))
    return loaded.forward(*inputs.args)


def attention_step_oracle(inputs: DecodeStepInputs) -> torch.Tensor:
    """What Hugging Face's own attention produces for the same drawn step."""
    return shape.decode_reference(inputs.layer, inputs.hidden_ctx, inputs.hidden_new)


def appended_cache_oracle(inputs: DecodeStepInputs) -> torch.Tensor:
    """The cache the step's caller should hold afterwards.

    Built the same way the input cache was, over the context with the decoded
    token appended: the kernel's returned latent is correct exactly when
    appending it reproduces this.
    """
    return shape.context_kv(
        inputs.layer, torch.cat([inputs.hidden_ctx, inputs.hidden_new], dim=1)
    )


__all__ = [
    "ACTIVATION_SEED",
    "CTX_LEN",
    "DEVICE",
    "DTYPE",
    "WEIGHT_SEED",
    "DecodeStepInputs",
    "appended_cache_oracle",
    "attention_step_inputs",
    "attention_step_oracle",
    "run_attention_step",
]
