"""The executable semantics one Qwen3.5-35B-A3B decode step is held to.

Inputs and oracle are one pair on purpose. The oracle is a Hugging Face layer
with random weights, so a factory that returned only tensors would leave the
reference free to score them against a differently initialised layer. What each
``*_step`` returns therefore carries both: what the kernels are handed, and the
module it was drawn from.

Weights are not arguments here. Each Module declares them ``ConstTensor``, so a
boundary is run by loading that Module from a ``DictResource`` keyed the way the
Module names its weights and then passing activations alone. Which Hugging Face
tensor a canonical name reads is this model's own fact, stated once in the
``*_constants`` mappings below.

Everything is seeded. The same call returns the same weights and the same
activations, so a disagreement is a disagreement about the compiler rather than
about which random draw each side happened to get.

Two things are deliberate about *how* the oracle is built.

**No Hugging Face cache object is constructed, on either side.** The step's
prior state is assembled as plain tensors, and the oracle's value is taken from a
forward over the whole sequence with the last position kept. Causality is what
makes the second equal a cached one-token step: the last position's output
depends on exactly the context before it. A reference that borrowed Hugging
Face's caching would be checking one cache implementation against another.

**Nothing is built that a boundary does not need.** A published MoE block holds
256 experts and is 3.2 GB in f32; a whole decoder layer is 3.3 GB, and the
published stack is forty of them. So a mixer boundary is given the mixer and its
norm and no MoE block at all, only the layer and MoE boundaries build a layer, and
everything is cached module-scoped so a worker builds each thing once. The
measured reason is in ``hf_layer``. There is no whole-model instantiation
anywhere in this package.

What is *not* covered here is listed in ``test_provenance.py`` rather than left
to be inferred.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tests.models import decode_oracle as oracle
from tests.models.decode_oracle import linear_weight
from tests.models.qwen3_5_35b_a3b.model import (
    LAYER_TYPE,
    MAX_CTX,
    Qwen3_5FullAttention,
    Qwen3_5LinearAttention,
    Qwen3_5MoE,
    published,
)
from tilefoundry.runtime.resource import DictResource

#: The checkpoint's own text configuration.
CONFIG = published()


#: The precision the checkpoint publishes; both sides of every comparison run at it.
DTYPE = CONFIG.dtype

#: Dimensions the published fields imply, named once where the oracles read them.
GDN_KEY_DIM = CONFIG.linear_num_key_heads * CONFIG.linear_key_head_dim
GDN_VALUE_DIM = CONFIG.linear_num_value_heads * CONFIG.linear_value_head_dim
#: Value heads sharing one key head.
GDN_V_PER_K = CONFIG.linear_num_value_heads // CONFIG.linear_num_key_heads
#: How many earlier positions the causal convolution needs: the kernel spans
#: `linear_conv_kernel_dim` positions ending at the one being decoded, so the
#: state handed in is the `kernel - 1` before it.
GDN_CONV_CONTEXT = CONFIG.linear_conv_kernel_dim - 1



def build_hf_decoder_layer(block_type: str, seed=0, device="cpu", dtype=DTYPE):
    """One ``Qwen3_5MoeDecoderLayer`` of *block_type*, weights drawn at *seed*.

    ``layer_idx`` is the lowest published index of that type, because the layer
    reads its own type out of ``config.layer_types[layer_idx]`` -- the index is
    how the type is chosen, not a decoration. The published config is passed
    whole: only one layer is constructed either way, so nothing has to shrink
    ``num_hidden_layers`` and ``layer_types`` together first.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeDecoderLayer,
    )

    index = CONFIG.layer_types.index(block_type)
    return oracle.randomised(
        lambda: Qwen3_5MoeDecoderLayer(CONFIG, layer_idx=index), seed, device, dtype
    )


def build_hf_mixer(block_type: str, seed=0, device="cpu", dtype=DTYPE):
    """One token mixer of *block_type* and the norm in front of it, and no more.

    A whole ``Qwen3_5MoeDecoderLayer`` is 1.7 GB at the published bf16, of which 3.2 GB is its
    256-expert MoE block -- and the mixer boundaries do not touch the MoE at all.
    Building the layer to test the mixer put that 3.2 GB in every parallel test
    worker at once and exhausted a 140 GB device when the rest of the suite ran
    alongside; measured, as an out-of-memory failure in eight of this package's
    tests under ``-n 8``. So the mixer is built on its own.

    The classes are the published ones -- ``Qwen3_5MoeAttention``,
    ``Qwen3_5MoeGatedDeltaNet``, ``Qwen3_5MoeRMSNorm`` -- at the published
    dimensions, held in a container that exposes them under the attribute names a
    decoder layer uses. So an oracle written against a layer reads this without
    knowing the difference, and what it is comparing against is still Hugging
    Face's own module rather than a reimplementation of it.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeAttention,
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeRMSNorm,
    )

    index = CONFIG.layer_types.index(block_type)

    class MixerOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = Qwen3_5MoeRMSNorm(
                CONFIG.hidden_size, eps=CONFIG.rms_norm_eps
            )
            if block_type == "linear_attention":
                self.linear_attn = Qwen3_5MoeGatedDeltaNet(CONFIG, index)
            else:
                self.self_attn = Qwen3_5MoeAttention(CONFIG, index)

    return oracle.randomised(MixerOnly, seed, device, dtype)


def rope_caches_at(total: int = 64, device="cpu", dtype=DTYPE):
    """cos / sin caches ``[total, rotary_dim]`` from the published rotary module.

    Narrower than ``head_dim``: ``partial_rotary_factor`` is 0.25, so the caches
    cover the 64 entries of each head that rotate and nothing else. That is what
    Hugging Face's own ``apply_rotary_pos_emb`` reads -- it slices ``q`` to
    ``cos.shape[-1]`` and concatenates the untouched tail back on.

    ``mrope`` is not exercised by a text-only fixture and this does not pretend
    it is. The published rotary embedding assigns a position triple per token and
    interleaves the three axes' frequencies by ``mrope_section``; with no image
    the three are the same number, so every branch of the interleave selects the
    same frequency and the result is ordinary RoPE at ``rotary_dim``. What these
    caches do cover is the partial factor. ``test_provenance.py`` measures the
    degeneracy rather than asserting it.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        Qwen3_5MoeTextRotaryEmbedding,
    )

    with torch.device(device):
        rotary = Qwen3_5MoeTextRotaryEmbedding(CONFIG)
    reference = torch.zeros(1, total, CONFIG.hidden_size, device=device)
    cos, sin = rotary(reference, torch.arange(total, device=device).unsqueeze(0))
    cos, sin = cos[0], sin[0]
    return (cos, sin) if dtype is None else (cos.to(dtype), sin.to(dtype))


def matrix_weight(weight):
    """A bare ``[out, in]`` parameter -> the kernels' ``[in, out]``.

    The MoE router and Hugging Face's expert tensors are ``nn.Parameter``s
    consumed by ``F.linear``, not ``nn.Linear`` modules, so they need the same
    transpose without the module wrapper.
    """
    return weight.t().contiguous()


#: The oracle's device. Every builder takes one; this is what the tests pass.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: Seeds, named so a change to either is a visible change to the reference.
WEIGHT_SEED = 0
ACTIVATION_SEED = 1

#: The context length a case is drawn at unless it states another. Small enough
#: to keep the oracle's full-sequence forward cheap, and coprime with both head
#: counts and with the convolution kernel, so an index arithmetic error cannot
#: coincide with a boundary.
CTX_LEN = 25


@functools.lru_cache(maxsize=None)
def hf_layer(block_type: str, device: str = DEVICE, whole_layer: bool = False):
    """The Hugging Face module a boundary of *block_type* is compared against.

    ``whole_layer`` picks how much of the layer is built, and it is a memory
    decision with a measured reason. A whole ``Qwen3_5MoeDecoderLayer`` is 3.3 GB
    in f32, essentially all of it the 256-expert MoE block; the mixer boundaries
    never touch that block, and building it for them put 3.2 GB in every parallel
    worker and ran a 140 GB device out of memory once the rest of the suite was
    alongside. So a mixer boundary gets the mixer and its norm, and only the two
    boundaries that genuinely span a layer get a layer.

    The two are *not* interchangeable: their parameters are drawn in construction
    order, and the orders differ, so the same seed gives different weights. Each
    boundary therefore takes both its arguments and its oracle from one object,
    which is what the ``*_step`` factories return.

    Cached per (type, device, extent). Safe to share -- eval mode, nothing writes.
    """
    if whole_layer:
        return build_hf_decoder_layer(
            block_type, seed=WEIGHT_SEED, device=device
        )
    return build_hf_mixer(block_type, seed=WEIGHT_SEED, device=device)


@functools.lru_cache(maxsize=None)
def rope_caches(device: str = DEVICE):
    """cos / sin caches covering every position a step may be decoded at."""
    return rope_caches_at(total=MAX_CTX, device=device)


def drawn_hidden(ctx_len: int, device: str = DEVICE):
    """A context of *ctx_len* tokens and the one token decoded after it."""
    torch.manual_seed(ACTIVATION_SEED)
    drawn = (torch.randn(1, ctx_len + 1, CONFIG.hidden_size, device=device) * 0.1).to(DTYPE)
    return drawn[:, :ctx_len], drawn[:, ctx_len:]


# ── what each Module holds, and where it is read from ───────────────────


def full_mixer_constants(layer) -> dict:
    """The full-attention mixer's weights, keyed the way its Module names them.

    ``w_qg`` is one projection with two jobs: Hugging Face's ``q_proj`` fans out
    to twice the query width, the second half being the output gate, so there is
    no separate gate tensor to read.
    """
    attention = layer.self_attn
    return {
        "gamma_in": layer.input_layernorm.weight,
        "w_qg": linear_weight(attention.q_proj),
        "w_k": linear_weight(attention.k_proj),
        "w_v": linear_weight(attention.v_proj),
        "gamma_q": attention.q_norm.weight,
        "gamma_k": attention.k_norm.weight,
        "w_o": linear_weight(attention.o_proj),
    }


def linear_mixer_constants(layer) -> dict:
    """The Gated DeltaNet's weights, keyed the way its Module names them."""
    mixer = layer.linear_attn
    return {
        "gamma_in": layer.input_layernorm.weight,
        "w_in_qkv": linear_weight(mixer.in_proj_qkv),
        "w_in_z": linear_weight(mixer.in_proj_z),
        "w_in_b": linear_weight(mixer.in_proj_b),
        "w_in_a": linear_weight(mixer.in_proj_a),
        # nn.Conv1d keeps a singleton in-channel axis a depthwise convolution
        # never uses; the kernel takes [channels, kernel].
        "conv_w": mixer.conv1d.weight.squeeze(1).contiguous(),
        "a_log": mixer.A_log,
        "dt_bias": mixer.dt_bias,
        # `Qwen3_5MoeRMSNormGated` scales by its weight directly, not by
        # 1 + weight the way the layer-level norms do.
        "gamma_gdn": mixer.norm.weight,
        "w_out": linear_weight(mixer.out_proj),
    }


def moe_constants(layer) -> dict:
    """*layer*'s MoE block weights, keyed the way ``moe`` names them.

    Hugging Face keeps the two SwiGLU halves in one ``gate_up_proj`` tensor, rows
    ``[:intermediate]`` the gate and ``[intermediate:]`` the up projection -- one
    ``F.linear`` then a chunk. Splitting them here is weight preprocessing, the
    same kind as transposing a projection, and it belongs on this side.
    """
    block = layer.mlp
    width = CONFIG.moe_intermediate_size
    gate_up = block.experts.gate_up_proj
    return {
        "gamma_post": layer.post_attention_layernorm.weight,
        "w_gate": gate_up[:, :width, :].contiguous(),
        "w_up": gate_up[:, width:, :].contiguous(),
        "w_down": block.experts.down_proj.contiguous(),
        "w_shared_gate": matrix_weight(block.shared_expert.gate_proj.weight),
        "w_shared_up": matrix_weight(block.shared_expert.up_proj.weight),
        "w_shared_down": matrix_weight(block.shared_expert.down_proj.weight),
        "w_shared_scale": matrix_weight(block.shared_expert_gate.weight),
    }


def router_constants(layer) -> dict:
    """*layer*'s router weight, keyed the way ``Qwen3_5Router`` names it."""
    return {"w_router": matrix_weight(layer.mlp.gate.weight)}


def moe_weights(layer) -> dict:
    """The MoE block's weights, each under the Module that declares it.

    The router is a child, so its weight is read under the child's own name --
    which is the same scoping ``load`` gives a nested Module anywhere else.
    """
    flat = dict(moe_constants(layer))
    flat.update({f"router.{name}": w for name, w in router_constants(layer).items()})
    return flat


#: The Module each published mixer kind is, and the mapping that fills it.
_MIXER = {
    "full_attention": (Qwen3_5FullAttention, full_mixer_constants),
    "linear_attention": (Qwen3_5LinearAttention, linear_mixer_constants),
}


def load_mixer(kind: str, layer):
    """The mixer Module of *kind* with *layer*'s weights bound.

    A loading, not a Module: the weights are read on the device *layer* holds
    them on, and the activations a caller then passes have to be on that one too.
    """
    module, constants = _MIXER[kind]
    return module.cloned().load(DictResource(constants(layer)))


def load_moe(layer):
    """The MoE Module with *layer*'s block weights bound, the router's included."""
    return Qwen3_5MoE.cloned().load(DictResource(moe_weights(layer)))


def load_layer(kind: str, layer):
    """The decoder layer Module of *kind* with *layer*'s weights bound.

    The keys are prefixed by the child each weight belongs to, which is how a
    layer's two blocks stay two namespaces: ``load`` scopes a child to its own
    subtree, so the mixer's ``gamma_in`` and the MoE's ``gamma_post`` are read
    under the names their own Modules use.
    """
    _module, constants = _MIXER[kind]
    flat = {f"mixer.{name}": w for name, w in constants(layer).items()}
    flat.update({f"moe.{name}": w for name, w in moe_weights(layer).items()})
    return LAYER_TYPE[kind].cloned().load(DictResource(flat))


def moe_oracle(layer, hidden) -> torch.Tensor:
    """What Hugging Face's own post-norm + MoE block produce for *hidden*."""
    with torch.no_grad():
        return layer.mlp(layer.post_attention_layernorm(hidden))


# ── full attention ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class FullStep:
    """One drawn full-attention decode step, and the layer behind it."""

    layer: object
    ctx_len: int
    hidden_ctx: torch.Tensor
    hidden_new: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    #: What the mixer is handed after the hidden state, in the order it declares
    #: them. Its weights are not here; a loading holds those.
    mixer_acts: tuple


def context_kv(layer, hidden, device: str = DEVICE):
    """The KV cache *layer* would hold for *hidden*, as explicit tensors.

    Built by running the layer's own norm, projections, key norm and rotary
    embedding over the context -- not approximately what its cache would hold,
    but the same tensors through the same modules in the same order.

    Returned in the kernels' ``[1, ctx_len, kv_heads, head_dim]`` layout, not
    Hugging Face's head-major one.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    length = hidden.shape[1]
    cos, sin = rope_caches(device)
    attention = layer.self_attn
    with torch.no_grad():
        normed = layer.input_layernorm(hidden)
        heads = (1, length, CONFIG.num_key_value_heads, CONFIG.head_dim)
        key = attention.k_norm(attention.k_proj(normed).view(heads)).transpose(1, 2)
        value = attention.v_proj(normed).view(heads).transpose(1, 2)
        _query, key = apply_rotary_pos_emb(
            key, key, cos[:length].unsqueeze(0), sin[:length].unsqueeze(0)
        )
    return key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()


def full_step(*, ctx_len: int = CTX_LEN, device: str = DEVICE,
              whole_layer: bool = False) -> FullStep:
    """One deterministic full-attention decode step over a *ctx_len* context.

    ``whole_layer`` builds the complete decoder layer, which is what the layer and
    MoE boundaries need; the mixer boundary leaves it off and pays neither the
    memory nor the build.
    """
    layer = hf_layer("full_attention", device, whole_layer)
    hidden_ctx, hidden_new = drawn_hidden(ctx_len, device)
    k_cache, v_cache = context_kv(layer, hidden_ctx, device)
    cos, sin = rope_caches(device)
    return FullStep(
        layer=layer,
        ctx_len=ctx_len,
        hidden_ctx=hidden_ctx,
        hidden_new=hidden_new,
        k_cache=k_cache,
        v_cache=v_cache,
        mixer_acts=(
            cos,
            sin,
            # The token being decoded sits immediately after the context.
            torch.tensor([ctx_len], device=device, dtype=torch.int32),
            k_cache,
            v_cache,
            torch.full(
                (1, 1, 1, 1), layer.self_attn.scaling, device=device, dtype=DTYPE
            ),
        ),
    )


def _whole_sequence(step) -> torch.Tensor:
    return torch.cat([step.hidden_ctx, step.hidden_new], dim=1)


def full_mixer_oracle(step: FullStep) -> torch.Tensor:
    """Hugging Face's own attention output at the decoded position.

    The whole sequence under a causal mask, last position kept. The mask exists
    only because Hugging Face's attention needs one for a multi-position
    forward; the kernel under test needs none, which is the point -- a single
    query at the end of the context may attend every position there is.
    """
    cos, sin = rope_caches(step.hidden_new.device.type)
    total = step.ctx_len + 1
    positions = torch.arange(total, device=step.hidden_new.device)
    mask = torch.where(
        positions.unsqueeze(0) <= positions.unsqueeze(1), 0.0, float("-inf")
    ).view(1, 1, total, total).to(step.hidden_new.dtype)
    with torch.no_grad():
        normed = step.layer.input_layernorm(_whole_sequence(step))
        out, _ = step.layer.self_attn(
            normed,
            position_embeddings=(cos[:total].unsqueeze(0), sin[:total].unsqueeze(0)),
            attention_mask=mask,
        )
    return out[:, -1:, :]


def full_layer_oracle(step: FullStep) -> torch.Tensor:
    """The complete Hugging Face decoder layer at the decoded position."""
    cos, sin = rope_caches(step.hidden_new.device.type)
    total = step.ctx_len + 1
    positions = torch.arange(total, device=step.hidden_new.device)
    mask = torch.where(
        positions.unsqueeze(0) <= positions.unsqueeze(1), 0.0, float("-inf")
    ).view(1, 1, total, total).to(step.hidden_new.dtype)
    with torch.no_grad():
        out = step.layer(
            _whole_sequence(step),
            position_embeddings=(cos[:total].unsqueeze(0), sin[:total].unsqueeze(0)),
            attention_mask=mask,
        )
    return out[:, -1:, :]


def run_full_attention_step(step: FullStep):
    """The full-attention mixer over *step*, weights bound from the layer drawn."""
    return load_mixer("full_attention", step.layer).full_attention(
        step.hidden_new, *step.mixer_acts
    )


def appended_cache_oracle(step: FullStep) -> tuple[torch.Tensor, torch.Tensor]:
    """The cache the step's caller should hold afterwards.

    Built the same way the input cache was, over the context with the decoded
    token appended: the kernel's returned key and value are correct exactly when
    appending them reproduces this.
    """
    return context_kv(step.layer, _whole_sequence(step), step.hidden_new.device.type)


# ── linear attention (Gated DeltaNet) ───────────────────────────────────


@dataclass(frozen=True)
class LinearStep:
    """One drawn Gated DeltaNet decode step, and the layer behind it."""

    layer: object
    ctx_len: int
    hidden_ctx: torch.Tensor
    hidden_new: torch.Tensor
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    #: What the mixer is handed after the hidden state, in the order it declares
    #: them: its state, and nothing else -- the rest it holds.
    mixer_acts: tuple


def run_linear_attention_step(step: LinearStep):
    """The Gated DeltaNet mixer over *step*, weights bound from the layer drawn."""
    return load_mixer("linear_attention", step.layer).linear_attention(
        step.hidden_new, *step.mixer_acts
    )


def gdn_state(layer, hidden) -> tuple[torch.Tensor, torch.Tensor]:
    """The state the Gated DeltaNet holds after *hidden*, as explicit tensors.

    Assembled out of the mixer's own modules and its own delta-rule function --
    ``in_proj_qkv``, ``conv1d``, ``in_proj_b``, ``in_proj_a`` and
    ``torch_chunk_gated_delta_rule`` -- in the order its ``forward`` applies
    them, with the ``Cache`` bookkeeping left out. That is the same construction
    the KV-cache layers get: state built by running the module over the context,
    not state copied out of a cache object.

    Two tensors come back:

    - the ``kernel - 1`` columns of the projection's output that the next
      position's causal convolution still needs. Hugging Face stores ``kernel``
      of them and drops the oldest on use; this drops it now.
    - the recurrent matrix after the context. Obtained from the chunked delta
      rule, which is the path a multi-position forward takes, so it is the same
      state the oracle's own forward passes through the decoded position.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
        torch_chunk_gated_delta_rule,
    )

    mixer = layer.linear_attn
    length = hidden.shape[1]
    window = GDN_CONV_CONTEXT
    with torch.no_grad():
        normed = layer.input_layernorm(hidden)
        projected = mixer.in_proj_qkv(normed).transpose(1, 2)
        conv_state = (
            projected[:, :, -window:]
            if length >= window
            else F.pad(projected, (window - length, 0))
        )
        # padding=kernel-1 on both sides, first `length` outputs kept: the
        # causal ones. Hugging Face's own prefill path, verbatim.
        convolved = F.silu(mixer.conv1d(projected)[:, :, :length]).transpose(1, 2)
        query, key, value = torch.split(
            convolved,
            [GDN_KEY_DIM, GDN_KEY_DIM, GDN_VALUE_DIM],
            dim=-1,
        )
        query = query.reshape(1, length, -1, CONFIG.linear_key_head_dim)
        key = key.reshape(1, length, -1, CONFIG.linear_key_head_dim)
        value = value.reshape(1, length, -1, CONFIG.linear_value_head_dim)
        query = query.repeat_interleave(GDN_V_PER_K, dim=2)
        key = key.repeat_interleave(GDN_V_PER_K, dim=2)
        beta = mixer.in_proj_b(normed).sigmoid()
        decay = -mixer.A_log.float().exp() * F.softplus(
            mixer.in_proj_a(normed).float() + mixer.dt_bias
        )
        _out, recurrent = torch_chunk_gated_delta_rule(
            query, key, value, g=decay, beta=beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    return conv_state.contiguous(), recurrent.contiguous()


def linear_step(*, ctx_len: int = CTX_LEN, device: str = DEVICE,
                whole_layer: bool = False) -> LinearStep:
    """One deterministic Gated DeltaNet decode step over a *ctx_len* context.

    ``whole_layer`` as in ``full_step``.
    """
    layer = hf_layer("linear_attention", device, whole_layer)
    hidden_ctx, hidden_new = drawn_hidden(ctx_len, device)
    conv_state, recurrent_state = gdn_state(layer, hidden_ctx)
    return LinearStep(
        layer=layer,
        ctx_len=ctx_len,
        hidden_ctx=hidden_ctx,
        hidden_new=hidden_new,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        mixer_acts=(conv_state, recurrent_state),
    )


def linear_mixer_oracle(step: LinearStep) -> torch.Tensor:
    """Hugging Face's own Gated DeltaNet output at the decoded position.

    A cache-free forward over the whole sequence, last position kept. The mixer
    is causal, so that position sees exactly the context before it -- the same
    argument the attention oracle rests on, for a mixer with a different kind of
    state.
    """
    with torch.no_grad():
        out = step.layer.linear_attn(
            step.layer.input_layernorm(_whole_sequence(step)), cache_params=None
        )
    return out[:, -1:, :]


def linear_layer_oracle(step: LinearStep) -> torch.Tensor:
    """The complete Hugging Face decoder layer at the decoded position."""
    with torch.no_grad():
        out = step.layer(_whole_sequence(step), position_embeddings=None)
    return out[:, -1:, :]


def advanced_state_oracle(step: LinearStep) -> tuple[torch.Tensor, torch.Tensor]:
    """The state the step's caller should hold afterwards.

    Built the same way the input state was, over the context with the decoded
    token appended.
    """
    conv, recurrent = gdn_state(step.layer, _whole_sequence(step))
    return conv.to(DTYPE), recurrent.to(DTYPE)


__all__ = [
    "ACTIVATION_SEED",
    "CTX_LEN",
    "DEVICE",
    "WEIGHT_SEED",
    "FullStep",
    "LinearStep",
    "advanced_state_oracle",
    "appended_cache_oracle",
    "context_kv",
    "drawn_hidden",
    "full_layer_oracle",
    "full_mixer_constants",
    "full_mixer_oracle",
    "full_step",
    "gdn_state",
    "hf_layer",
    "linear_layer_oracle",
    "linear_mixer_constants",
    "linear_mixer_oracle",
    "linear_step",
    "load_layer",
    "load_mixer",
    "load_moe",
    "moe_constants",
    "moe_oracle",
    "rope_caches",
    "run_full_attention_step",
    "run_linear_attention_step",
]
