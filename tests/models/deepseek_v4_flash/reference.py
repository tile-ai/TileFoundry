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

from tests.models.deepseek_v4_flash.model import (
    FP8E4M3_MAX,
    FP8E4M3_QUANT_EPS,
    HF_CONFIG,
    KV_QUANT_BLOCK,
    REAL,
    DeepseekV4Attention,
    DSV4Config,
)
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
CTX_LEN = REAL.window - 1


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
    real = REAL
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
    real = REAL
    layer = build_hf_attention(seed=WEIGHT_SEED, device=device, dtype=DTYPE)

    torch.manual_seed(ACTIVATION_SEED)
    drawn = (torch.randn(1, ctx_len + 1, real.dim, device=device) * 0.1).to(DTYPE)
    hidden_ctx, hidden_new = drawn[:, :ctx_len], drawn[:, ctx_len:]
    kv_cache = context_kv(layer, hidden_ctx)

    # The token being decoded sits immediately after the context.
    cos, sin = rope_caches(ctx_len + 1, device)
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
    return decode_reference(inputs.layer, inputs.hidden_ctx, inputs.hidden_new)


def appended_cache_oracle(inputs: DecodeStepInputs) -> torch.Tensor:
    """The cache the step's caller should hold afterwards.

    Built the same way the input cache was, over the context with the decoded
    token appended: the kernel's returned latent is correct exactly when
    appending it reproduces this.
    """
    return context_kv(
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


#: The first layer the published config declares as sliding, so a test that
#: means "a sliding layer" names one the checkpoint agrees is one.
SLIDING_LAYER = HF_CONFIG.layer_types.index("sliding_attention")


def small() -> DSV4Config:
    """The same model, small enough to run end to end.

    Every dimension stays divisible by ``quant_block`` / ``KV_QUANT_BLOCK``,
    as the real shape rules require. Test-side because it is nobody's
    checkpoint: the published shape is what `model.published()` reads.
    """
    return DSV4Config(
        dim=256,
        n_heads=2,
        n_kv_heads=1,
        head_dim=256,
        rope_dim=64,
        q_lora_rank=128,
        o_groups=2,
        o_lora_rank=128,
        window=8,
        vocab=64,
        moe_inter=256,
        n_routed=4,
        n_act=2,
        route_scale=1.5,
        swiglu_limit=10.0,
        n_layers=1,
        n_hash_layers=1,
        rms_eps=1e-6,
        quant_block=128,
        compress_ratios=(0,),
    )


def fake_quant_kv(latent):
    """*latent*'s unrotated head dims, through the checkpoint's fp8 KV round trip.

    The one place a reference for this model is not Hugging Face's own module.
    V4-Flash stores its KV latent as fp8 e4m3 with a per-block power-of-two
    (ue8m0) scale, which the official inference path does and
    ``modeling_deepseek_v4`` does not; a reference without it would be scoring
    the kernel against a model that keeps precision the kernel is specified to
    throw away. Only the unrotated dims are quantized -- the rope slice stays
    bf16, as it does in the kernel.
    """
    import torch  # noqa: PLC0415

    nope, rope = latent[..., : REAL.nope_dim], latent[..., REAL.nope_dim :]
    blocks = nope.float().reshape(*nope.shape[:-1], REAL.kv_quant_blocks, KV_QUANT_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(FP8E4M3_QUANT_EPS)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8E4M3_MAX)))
    scaled = (blocks / scale).clamp(-FP8E4M3_MAX, FP8E4M3_MAX)
    dequant = scaled.to(torch.float8_e4m3fn).to(torch.float32) * scale
    return torch.cat([dequant.reshape(nope.shape).to(latent.dtype), rope], dim=-1)


def build_hf_attention(seed=0, device="cuda", dtype=None):
    """A ``DeepseekV4Attention`` for the sliding layer, random at a fixed seed.

    The fp8 KV round trip is installed on the module's own KV norm rather than
    applied by whoever calls it, so every path that reads this layer's stored
    latent -- the cache built from a context, and the full-sequence forward the
    oracle takes its answer from -- stores the same thing the kernel does.
    """
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
        DeepseekV4Attention,
    )

    from tests.models import decode_oracle as oracle  # noqa: PLC0415

    layer = oracle.randomised(
        lambda: DeepseekV4Attention(HF_CONFIG, layer_idx=SLIDING_LAYER),
        seed, device, dtype,
    )
    layer.kv_norm.register_forward_hook(lambda _m, _args, out: fake_quant_kv(out))
    return layer


def rope_caches(total: int, device="cuda"):
    """Interleaved cos / sin ``[1, total, rope_half]`` for the ``main`` rope label.

    One entry per rotated pair, which is what V4's interleaved rotation takes --
    and why ``decode_oracle.rope_caches`` cannot build it: this model's rotary
    embedding is keyed by layer type and returns the half-width pair, not the
    duplicated full-width one every other model in the corpus uses.
    """
    import torch  # noqa: PLC0415
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
        DeepseekV4RotaryEmbedding,
    )

    rotary = DeepseekV4RotaryEmbedding(HF_CONFIG).to(device)
    reference = torch.zeros(1, total, REAL.dim, device=device)
    positions = torch.arange(total, device=device).unsqueeze(0)
    return rotary(reference, positions, layer_type="main")


def context_kv(layer, hidden_ctx):
    """The KV cache *layer* would hold for *hidden_ctx*, as an explicit tensor.

    Its own norm, its own projection, its own rotation, in its own order -- the
    same tensors its cache would have held, in the kernels'
    ``[1, ctx_len, n_kv_heads, head_dim]`` layout rather than Hugging Face's
    head-major one. ``decode_oracle.context_kv`` states this for a model with a
    separate key and value drawn through a pair-wise rotary; V4 has one shared
    latent and a single-tensor rotation, so the shape of that hook does not fit.
    """
    import torch  # noqa: PLC0415
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    ctx_len = hidden_ctx.shape[1]
    cos, sin = rope_caches(ctx_len, hidden_ctx.device.type)
    with torch.no_grad():
        latent = layer.kv_norm(layer.kv_proj(hidden_ctx))
        latent = latent.view(1, ctx_len, 1, REAL.head_dim).transpose(1, 2)
        rotated = apply_rotary_pos_emb(latent, cos.to(latent.dtype), sin.to(latent.dtype))
    return rotated.transpose(1, 2).contiguous()


def decode_reference(layer, hidden_ctx, hidden_new):
    """What *layer* produces for *hidden_new* decoded after *hidden_ctx*.

    The whole sequence under a causal mask, last position kept, no ``Cache``
    object on either side. Causal is the whole story only while the sequence
    fits the window, which is why the context is asked to.
    """
    import torch  # noqa: PLC0415

    from tests.models import decode_oracle as oracle  # noqa: PLC0415

    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    if total > REAL.window:
        raise ValueError(
            f"a sliding layer attends {REAL.window} positions counting its own; "
            f"a {total}-long sequence is not a decode step it can take"
        )
    device = hidden_ctx.device.type
    cos, sin = rope_caches(total, device)
    mask = oracle.causal_mask(total, device, hidden_ctx.dtype)
    positions = torch.arange(total, device=device).unsqueeze(0)
    with torch.no_grad():
        out, _ = layer(
            torch.cat([hidden_ctx, hidden_new], dim=1),
            position_embeddings={"main": (cos.to(hidden_ctx.dtype), sin.to(hidden_ctx.dtype))},
            position_ids=positions,
            attention_mask=mask,
        )
    return out[:, hidden_ctx.shape[1] :, :]


__all__ = [
    "FP8E4M3_MAX",
    "FP8E4M3_QUANT_EPS",
    "HF_CONFIG",
    "KV_QUANT_BLOCK",
    "REAL",
    "SLIDING_LAYER",
    "small",
    "DSV4Config",
    "build_hf_attention",
    "context_kv",
    "decode_reference",
    "fake_quant_kv",
    "rope_caches",
]
