"""Qwen3.5-35B-A3B's published submodules and both of its decoder layer types.

The two layer types differ only in which token mixer they hold. Each is its own
Module rather than one Module with a branch, because they are different kernels:
a branch would give analysis and scheduling one domain for two behaviours.
They share the walk, which composes Modules and so cannot be a ``@func``.
"""
from __future__ import annotations

import math

from tests.models.qwen3_5_35b_a3b.config import REAL as config
from tilefoundry import DType, func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: F401 -- tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings
from tilefoundry.evaluator import to_torch_dtype
from tilefoundry.ir.types.dim import DimVar

# One token per step. No other extent here is dynamic either: this mixer's state
# is fixed-size, so the module carries no DimVar.
S = 1

_H = config.hidden
_HK = config.gdn_n_k_heads
_HV = config.gdn_n_v_heads
_DK = config.gdn_head_k_dim
_DV = config.gdn_head_v_dim
_KEY = config.gdn_key_dim
_VAL = config.gdn_value_dim
_CONV = config.gdn_conv_dim
_KERNEL = config.gdn_conv_kernel
_WINDOW = config.gdn_conv_context
_VPK = config.gdn_v_per_k

# The delta rule's query scale, and the epsilon its L2 normalisation uses. Both
# are architecture constants rather than runtime values -- they are fixed by
# ``linear_key_head_dim``, not chosen per step -- so they are folded in here
# instead of taking up parameters a caller would have to get right.
_QSCALE = 1.0 / math.sqrt(config.gdn_head_k_dim)
_L2_EPS = 1e-6


# The active context length: the only range this kernel carries. DimVar bounds
# are half-open [lo, hi), so the envelope's exclusive upper bound is max_ctx + 1
# to keep the largest supported context inside it. The lower bound is 1: a step
# with no prior context is a prefill, not a decode step.
C = DimVar("ctx_len", 1, config.max_ctx + 1)

# One token per step.

_HQ = config.n_q_heads
_HKV = config.n_kv_heads
_D = config.head_dim
_ROT = config.rotary_dim
_PASS = config.pass_dim
_G = config.gqa_group

# The rotary caches need one row per position a step may be decoded at, which is
# at most ``max_ctx``; ``max_position_embeddings`` is 262144 and a cache that
# size is 67 MB of zeros nothing reads.
_ROPE_ROWS = config.max_ctx + 1


# One token per step.

_E = config.n_experts
_K = config.top_k
_I = config.moe_intermediate
_IS = config.shared_intermediate


@module(entry="linear_attention")
class Qwen3_5LinearAttention:
    @func
    def conv_step(
        conv_state: Tensor[(1, _CONV, _WINDOW), config.dt],
        entry: Tensor[(1, _CONV, S), config.dt],
        conv_w: ConstTensor[(_CONV, _KERNEL), config.dt],
    ) -> Tensor[(1, _CONV), config.dt]:
        # The depthwise causal convolution at one token per step: the window
        # closes on this token, so the whole convolution is one multiply against
        # the kernel and one reduction over it. Channels do not mix -- that is
        # what depthwise means here, and it is why no matmul appears.
        window = tf.concat(conv_state, entry, axis=2)
        weighted = window * tf.reshape(conv_w, new_shape=(1, _CONV, _KERNEL))
        summed = tf.reduce(weighted, axes=(-1,), keepdim=False, kind="sum")
        return tf.silu(summed)

    @func
    def l2_normalise(
        x: Tensor[(1, S, _HV, _DK), config.dt],
    ) -> Tensor[(1, S, _HV, _DK), config.dt]:
        # Per-head L2 normalisation, matching the linear-attention library's own
        # (`l2norm` in the Hugging Face module): rsqrt of the *sum* of squares
        # plus eps, not of the mean, so it is not an RMSNorm with a unit scale.
        square_sum = tf.reduce(tf.square(x), axes=(-1,), keepdim=True, kind="sum")
        return x * tf.rsqrt(square_sum + tf.full_like(square_sum, value=_L2_EPS))

    @func
    def delta_step(
        recurrent_state: Tensor[(1, _HV, _DK, _DV), config.dt],
        q: Tensor[(1, S, _HV, _DK), config.dt],
        k: Tensor[(1, S, _HV, _DK), config.dt],
        v: Tensor[(1, S, _HV, _DV), config.dt],
        g: Tensor[(1, S, _HV), config.dt],
        beta: Tensor[(1, S, _HV), config.dt],
    ):
        # One token of the gated delta rule. Returns the read-out and the updated
        # state, in that order; the state is an output because a rank-one update
        # has no smaller increment to hand back.
        decayed = recurrent_state * tf.reshape(tf.exp(g), new_shape=(1, _HV, 1, 1))
        k_col = tf.reshape(k, new_shape=(1, _HV, _DK, 1))
        recalled = tf.reduce(decayed * k_col, axes=(-2,), keepdim=False, kind="sum")
        delta = (tf.reshape(v, new_shape=(1, _HV, _DV)) - recalled) * tf.reshape(
            beta, new_shape=(1, _HV, 1)
        )
        updated = decayed + k_col * tf.reshape(delta, new_shape=(1, _HV, 1, _DV))
        q_scaled = q * tf.full_like(q, value=_QSCALE)
        read = tf.reduce(
            updated * tf.reshape(q_scaled, new_shape=(1, _HV, _DK, 1)),
            axes=(-2,), keepdim=False, kind="sum",
        )
        return read, updated

    @func
    def linear_attention(
        hidden: Tensor[(1, S, _H), config.dt],
        gamma_in: ConstTensor[(_H,), config.dt],
        w_in_qkv: ConstTensor[(1, _H, _CONV), config.dt],
        w_in_z: ConstTensor[(1, _H, _VAL), config.dt],
        w_in_b: ConstTensor[(1, _H, _HV), config.dt],
        w_in_a: ConstTensor[(1, _H, _HV), config.dt],
        conv_w: ConstTensor[(_CONV, _KERNEL), config.dt],
        a_log: ConstTensor[(_HV,), config.dt],
        dt_bias: ConstTensor[(_HV,), config.dt],
        conv_state: Tensor[(1, _CONV, _WINDOW), config.dt],
        recurrent_state: Tensor[(1, _HV, _DK, _DV), config.dt],
        gamma_gdn: ConstTensor[(_DV,), config.dt],
        w_out: ConstTensor[(1, _VAL, _H), config.dt],
    ):
        # Fused input_layernorm + `Qwen3_5MoeGatedDeltaNet`, no residual (the
        # layer owns the residual add). Returns the output, this step's own
        # convolution column, and the updated recurrent state.
        hidden_norm = tf.rms_norm(hidden, gamma_in)

        entry = tf.transpose(tf.matmul(hidden_norm, w_in_qkv), perm=(0, 2, 1))
        mixed = conv_step(conv_state, entry, conv_w)

        q_flat = mixed[:, :_KEY]
        k_flat = mixed[:, _KEY : 2 * _KEY]
        v_flat = mixed[:, 2 * _KEY : _CONV]

        # Every value head reads the key head it shares; the projection produces
        # one key head per group, and the delta rule runs per value head.
        q = l2_normalise(
            tf.repeat_interleave(
                tf.reshape(q_flat, new_shape=(1, S, _HK, _DK)), repeats=_VPK, axis=2
            )
        )
        k = l2_normalise(
            tf.repeat_interleave(
                tf.reshape(k_flat, new_shape=(1, S, _HK, _DK)), repeats=_VPK, axis=2
            )
        )
        v = tf.reshape(v_flat, new_shape=(1, S, _HV, _DV))

        beta = tf.sigmoid(tf.matmul(hidden_norm, w_in_b))
        # g is negative by construction, so exp(g) is a decay in (0, 1): the
        # state cannot grow without a token asking for it through the rank-one
        # update.
        g = -tf.exp(a_log) * tf.softplus(tf.matmul(hidden_norm, w_in_a) + dt_bias)

        read, updated = delta_step(recurrent_state, q, k, v, g, beta)

        # The gated output norm: normalise per value head, scale, then gate by a
        # projection of the layer input through silu.
        z = tf.reshape(tf.matmul(hidden_norm, w_in_z), new_shape=(1, _HV, _DV))
        normed = tf.rms_norm(read, gamma_gdn)
        gated = normed * tf.silu(z)
        out = tf.matmul(tf.reshape(gated, new_shape=(1, S, _VAL)), w_out)
        return out, entry, updated

@module(entry="full_attention")
class Qwen3_5FullAttention:
    @func
    def partial_rope(
        x: Tensor[(1, S, _HQ, _D), config.dt],
        cos_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        sin_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        pos_ids: Tensor[(S,), "i32"],
    ) -> Tensor[(1, S, _HQ, _D), config.dt]:
        # Rotate the leading `rotary_dim` of each head and concatenate the
        # untouched tail back on. `tf.rope` multiplies its caches against the
        # whole of its input's last axis, so the split is what makes a partial
        # factor expressible at all rather than an optional rearrangement.
        rot = x[:, :, :, :_ROT]
        tail = x[:, :, :, _ROT:_D]
        turned, _ = tf.rope(rot, rot, cos_cache, sin_cache, pos_ids)
        return tf.concat(turned, tail, axis=-1)

    @func
    def partial_rope_kv(
        x: Tensor[(1, S, _HKV, _D), config.dt],
        cos_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        sin_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        pos_ids: Tensor[(S,), "i32"],
    ) -> Tensor[(1, S, _HKV, _D), config.dt]:
        # The same rotation over the key's head count. Its own Function because a
        # Function's parameter shapes are fixed and GQA's two head counts differ.
        rot = x[:, :, :, :_ROT]
        tail = x[:, :, :, _ROT:_D]
        turned, _ = tf.rope(rot, rot, cos_cache, sin_cache, pos_ids)
        return tf.concat(turned, tail, axis=-1)

    @func
    def full_attention(
        hidden: Tensor[(1, S, _H), config.dt],
        gamma_in: ConstTensor[(_H,), config.dt],
        w_qg: ConstTensor[(1, _H, _HQ * _D * 2), config.dt],
        w_k: ConstTensor[(1, _H, _HKV * _D), config.dt],
        w_v: ConstTensor[(1, _H, _HKV * _D), config.dt],
        gamma_q: ConstTensor[(_D,), config.dt],
        gamma_k: ConstTensor[(_D,), config.dt],
        cos_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        sin_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, _HKV, _D), config.dt],
        v_cache: Tensor[(1, C, _HKV, _D), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: ConstTensor[(1, _HQ * _D, _H), config.dt],
    ):
        # Fused input_layernorm + `Qwen3_5MoeAttention`, no residual (the layer
        # owns the residual add). Returns the attention output together with this
        # token's key and value, which are what the caller appends to the cache.
        hidden_norm = tf.rms_norm(hidden, gamma_in)

        # One projection, two halves: the query and the output gate. The split is
        # over the last axis of the [heads, 2 * head_dim] view, so gate entry j of
        # head h sits beside query entry j of the same head, not in a second
        # contiguous block of the flat projection.
        qg = tf.reshape(
            tf.matmul(hidden_norm, w_qg), new_shape=(1, S, _HQ, 2 * _D)
        )
        q = qg[:, :, :, :_D]
        gate = qg[:, :, :, _D : 2 * _D]

        q_rope = partial_rope(
            tf.rms_norm(q, gamma_q), cos_cache, sin_cache, pos_ids
        )
        k_rope = partial_rope_kv(
            tf.rms_norm(
                tf.reshape(tf.matmul(hidden_norm, w_k), new_shape=(1, S, _HKV, _D)),
                gamma_k,
            ),
            cos_cache, sin_cache, pos_ids,
        )
        v = tf.reshape(tf.matmul(hidden_norm, w_v), new_shape=(1, S, _HKV, _D))

        # Every query head sees its group's key/value head, for the cache and for
        # the new token alike.
        q_s = q_rope * scale
        k_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(k_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, _HQ, C, _D),
        )
        v_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(v_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, _HQ, C, _D),
        )
        k_new = tf.repeat_interleave(k_rope, repeats=_G, axis=2)
        v_new = tf.repeat_interleave(v, repeats=_G, axis=2)

        # Two score groups: one over the cache, one over the token itself.
        q_e = tf.reshape(q_s, new_shape=(1, S, _HQ, 1, _D))
        score_ctx = tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=True, kind="sum")
        score_new = tf.reduce(q_s * k_new, axes=(-1,), keepdim=True, kind="sum")

        # Log-sum-exp merge of the two groups' partials against their joint max.
        peak = tf.max(
            tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
        )
        peak_e = tf.reshape(peak, new_shape=(1, S, _HQ, 1, 1))
        p_ctx = tf.exp(score_ctx - peak_e)
        p_new = tf.exp(score_new - peak)
        total = tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum") + p_new
        weighted = (
            tf.reduce(p_ctx * v_ctx, axes=(-2,), keepdim=False, kind="sum")
            + p_new * v_new
        )
        attn = weighted / total

        # The output gate, then o_proj. Head-major flattening on both sides, so
        # gate entry (h, j) meets attention entry (h, j).
        gated = tf.reshape(attn, new_shape=(1, S, _HQ * _D)) * tf.sigmoid(
            tf.reshape(gate, new_shape=(1, S, _HQ * _D))
        )
        return tf.matmul(gated, w_o), k_rope, v

@module(entry="moe")
class Qwen3_5MoE:
    @func
    def routing(
        tokens: Tensor[(S, _H), config.dt],
        w_router: ConstTensor[(_H, _E), config.dt],
    ):
        # HF `Qwen3_5MoeTopKRouter`: softmax over every expert in f32, then the
        # top k, then renormalise. Kept its own function because the selection
        # is the one value in this block that is an index rather than a number,
        # and a router that picked a different 8 would be a different model even
        # if every weight matched.
        logits = tf.cast(tf.matmul(tokens, w_router), dtype="f32")
        probs = tf.softmax(logits, axis=-1)
        top_vals, indices = tf.topk(probs, k=_K, axis=-1)
        denom = tf.reduce(top_vals, axes=(-1,), keepdim=True, kind="sum")
        return tf.cast(top_vals / denom, dtype=config.dt), indices

    @func
    def routed_experts(
        tokens: Tensor[(S, _H), config.dt],
        weights: Tensor[(S, _K), config.dt],
        indices: Tensor[(S, _K), "i64"],
        w_gate: ConstTensor[(_E, _I, _H), config.dt],
        w_up: ConstTensor[(_E, _I, _H), config.dt],
        w_down: ConstTensor[(_E, _H, _I), config.dt],
    ) -> Tensor[(S, _H), config.dt]:
        # The gathers are the point: `indices` is a runtime value, so the three
        # expert tensors are indexed by it rather than sliced at a known offset.
        # Each token then runs `top_k` independent SwiGLU experts, batched over
        # the (token, slot) pair, and their outputs are mixed by the routing
        # weights.
        gate_w = tf.gather(w_gate, indices, axis=0)
        up_w = tf.gather(w_up, indices, axis=0)
        down_w = tf.gather(w_down, indices, axis=0)
        token_col = tf.reshape(tokens, new_shape=(S, 1, _H, 1))
        gate = tf.reshape(tf.matmul(gate_w, token_col), new_shape=(S, _K, _I))
        up = tf.reshape(tf.matmul(up_w, token_col), new_shape=(S, _K, _I))
        hidden = tf.silu(gate) * up
        down = tf.reshape(
            tf.matmul(down_w, tf.reshape(hidden, new_shape=(S, _K, _I, 1))),
            new_shape=(S, _K, _H),
        )
        weighted = down * tf.reshape(weights, new_shape=(S, _K, 1))
        return tf.reduce(weighted, axes=(1,), keepdim=False, kind="sum")

    @func
    def shared_expert(
        tokens: Tensor[(S, _H), config.dt],
        w_shared_gate: ConstTensor[(_H, _IS), config.dt],
        w_shared_up: ConstTensor[(_H, _IS), config.dt],
        w_shared_down: ConstTensor[(_IS, _H), config.dt],
        w_shared_scale: ConstTensor[(_H, 1), config.dt],
    ) -> Tensor[(S, _H), config.dt]:
        # A dense SwiGLU every token goes through, scaled by the token's own
        # scalar gate. The gate is a projection to width one through a sigmoid,
        # so it is between 0 and 1 per token and cannot change sign.
        gate = tf.matmul(tokens, w_shared_gate)
        up = tf.matmul(tokens, w_shared_up)
        dense = tf.matmul(tf.silu(gate) * up, w_shared_down)
        scale = tf.sigmoid(tf.matmul(tokens, w_shared_scale))
        return dense * scale

    @func
    def moe(
        hidden: Tensor[(1, S, _H), config.dt],
        gamma_post: ConstTensor[(_H,), config.dt],
        w_router: ConstTensor[(_H, _E), config.dt],
        w_gate: ConstTensor[(_E, _I, _H), config.dt],
        w_up: ConstTensor[(_E, _I, _H), config.dt],
        w_down: ConstTensor[(_E, _H, _I), config.dt],
        w_shared_gate: ConstTensor[(_H, _IS), config.dt],
        w_shared_up: ConstTensor[(_H, _IS), config.dt],
        w_shared_down: ConstTensor[(_IS, _H), config.dt],
        w_shared_scale: ConstTensor[(_H, 1), config.dt],
    ) -> Tensor[(1, S, _H), config.dt]:
        # Fused post_attention_layernorm + `Qwen3_5MoeSparseMoeBlock`, no
        # residual (the layer owns the residual add).
        tokens = tf.reshape(tf.rms_norm(hidden, gamma_post), new_shape=(S, _H))
        weights, indices = routing(tokens, w_router)
        routed = routed_experts(tokens, weights, indices, w_gate, w_up, w_down)
        shared = shared_expert(
            tokens, w_shared_gate, w_shared_up, w_shared_down, w_shared_scale
        )
        return tf.reshape(routed + shared, new_shape=(1, S, _H))


def _layer_forward(self, hidden, mixer_args):
    """One decode step: mixer + residual, then MoE + residual.

    Mirrors ``Qwen3_5MoeDecoderLayer.forward``. The two pre-norms are not
    here because each block fuses its own -- the mixer fuses
    ``input_layernorm`` and the MoE block fuses
    ``post_attention_layernorm``, so each fused kernel lines up with one
    Hugging Face pre-norm-then-block composition.

    *mixer_args* is what the mixer is handed after the hidden state. The MoE
    block is handed the mixed state and nothing else: every weight it reads is
    one it holds.

    What comes back is the layer output and whatever state the mixer
    produced, passed through untouched for the caller to advance.
    """
    mixed, *state = self.mixer(hidden, *mixer_args)
    attended = self.residual_add(hidden, mixed)
    expert_out = self.moe(attended)
    return self.residual_add(attended, expert_out), tuple(state)


@module(entry="residual_add")
class Qwen3_5FullAttnLayer:
    mixer = Qwen3_5FullAttention.renamed("mixer")
    moe = Qwen3_5MoE.renamed("moe")

    @func
    def residual_add(
        a: Tensor[(1, S, config.hidden), config.dt],
        b: Tensor[(1, S, config.hidden), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        return a + b

    forward = _layer_forward


@module(entry="residual_add")
class Qwen3_5LinearAttnLayer:
    mixer = Qwen3_5LinearAttention.renamed("mixer")
    moe = Qwen3_5MoE.renamed("moe")

    @func
    def residual_add(
        a: Tensor[(1, S, config.hidden), config.dt],
        b: Tensor[(1, S, config.hidden), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        return a + b

    forward = _layer_forward


#: Which layer class each published `layer_types` entry names. The model states
#: this, not its tests: it is the same fact `config.layer_types` is written in.
LAYER_TYPE = {
    "full_attention": Qwen3_5FullAttnLayer,
    "linear_attention": Qwen3_5LinearAttnLayer,
}

#: config.dt as torch spells it -- the state below is at the kernels' own dtype.
_TORCH_DT = to_torch_dtype(DType.from_name(config.dt))


#: The parameters a mixer declares for its own state, whichever kind it is. The
#: root splices a layer's cache in at the first of them.
_CACHE_PARAMS = frozenset({"k_cache", "v_cache", "conv_state", "recurrent_state"})


def _with_cache(mixer, mixer_args, cache):
    """*mixer_args* with *cache* spliced in where *mixer* declares its state.

    The position is counted over the parameters a step is handed, since a loading
    fills the weights by name, and read from the Module a loading stands over so
    that one rule answers for both.
    """
    node = getattr(mixer, "module", mixer)
    names = [
        param.name for param in node.entry_function().params if not param.is_const
    ][1:]
    # `next`, not `min`: `from tilefoundry.dsl.tf import *` binds `min` to the op.
    at = next(index for index, name in enumerate(names) if name in _CACHE_PARAMS)
    return (*mixer_args[:at], *cache, *mixer_args[at:])


def advance_state(kind, state, fresh):
    """A layer of *kind*'s next state, from what its mixer returned.

    The recurrent matrix is replaced whole -- a rank-one update has no smaller
    increment -- while the convolution window slides by the one column the step
    produced. Key and value are appended.
    """
    import torch  # noqa: PLC0415

    if kind == "linear_attention":
        window, _matrix = state
        column, updated = fresh
        return torch.cat([window, column], dim=2)[:, :, -_WINDOW:], updated
    return tuple(torch.cat([old, new], dim=1) for old, new in zip(state, fresh))


@module
class Qwen3_5Decoder:
    """The layer stack in `config.layer_types` order, and the step around it --
    embedding, the walk, the closing norm, the head. Each layer is an independent
    copy, so an analysis of one annotates only it."""

    layers = tuple(
        LAYER_TYPE[kind].renamed(f"layer{index}")
        for index, kind in enumerate(config.layer_types)
    )

    @func
    def embed(
        table: ConstTensor[(config.vocab, config.hidden), config.dt],
        token_ids: Tensor[(1,), "i64"],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # HF `Qwen3_5MoeModel.embed_tokens`: the decoded token's own row.
        return tf.reshape(
            tf.gather(table, token_ids, axis=0), new_shape=(1, S, config.hidden)
        )

    @func
    def final_rms_norm(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_final: ConstTensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # HF `Qwen3_5MoeModel.norm`, applied once after the last layer.
        return tf.rms_norm(hidden, gamma_final, eps=config.rms_eps)

    @func
    def lm_head(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        w_head: ConstTensor[(config.hidden, config.vocab), config.dt],
    ) -> Tensor[(1, config.vocab), config.dt]:
        # HF `Qwen3_5MoeForCausalLM.lm_head`, over the one token being decoded.
        return tf.matmul(tf.reshape(hidden, new_shape=(1, config.hidden)), w_head)

    @lm_head.converter("w_head")
    def _(
        head_weight_raw: ConstTensor[(config.vocab, config.hidden), config.dt],
    ) -> Tensor[(config.hidden, config.vocab), config.dt]:
        # HF stores the head as (vocab, hidden); the matmul above wants it the
        # other way. Tied models alias this input to the embedding table.
        return tf.transpose(head_weight_raw, perm=(1, 0))

    def decode_hidden(self, hidden, layer_args, caches):
        """One decode step through every layer, then the final norm.

        *layer_args* is one layer's mixer arguments per layer, carrying no state;
        *caches* is each layer's own state, spliced into its mixer call. What
        comes back is the normed hidden state and each layer's fresh state.
        """
        if len(layer_args) != len(self.modules) or len(caches) != len(self.modules):
            raise ValueError(
                f"decoder has {len(self.modules)} layers but was given "
                f"{len(layer_args)} argument tuples and {len(caches)} caches"
            )
        states = []
        for layer, mixer_args, cache in zip(self.modules, layer_args, caches):
            hidden, state = layer(hidden, _with_cache(layer.mixer, mixer_args, cache))
            states.append(state)
        return self.final_rms_norm(hidden), tuple(states)

    def forward(self, token_ids, layer_args, caches):
        """One decode step of the whole model: token ids in, logits out.

        The fresh per-layer state comes out beside the logits; growing *caches*
        with it is the caller's step, through `append_cache`.
        """
        hidden = self.embed(token_ids)
        normed, states = self.decode_hidden(hidden, layer_args, caches)
        return self.lm_head(normed), states

    def init_caches(self, device="cuda"):
        """The per-layer state container, one entry per layer.

        A linear-attention layer's two halves are genuinely zero at the start:
        Hugging Face left-pads the convolution window when the context is shorter
        than it, and `initial_state=None` is the zero recurrent matrix. An
        attention layer gets a container of no positions -- not a decode start,
        since no prefix produced it and `ctx_len` is bounded below by 1, so
        `forward` needs a context the caller prefilled.
        """
        import torch  # noqa: PLC0415

        entries = []
        for kind in config.layer_types:
            if kind == "linear_attention":
                entries.append((
                    torch.zeros(1, _CONV, _WINDOW, dtype=_TORCH_DT, device=device),
                    torch.zeros(1, _HV, _DK, _DV, dtype=_TORCH_DT, device=device),
                ))
            else:
                empty = torch.zeros(1, 0, _HKV, _D, dtype=_TORCH_DT, device=device)
                entries.append((empty, empty))
        return tuple(entries)

    def append_cache(self, caches, fresh):
        """Every layer's state advanced by the step it just took: a kernel hands
        back its own token's entry, and joining it on is the caller's."""
        return tuple(
            advance_state(kind, cache, new)
            for kind, cache, new in zip(config.layer_types, caches, fresh)
        )
