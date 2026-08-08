"""Granite-4.0-H-Small as authored TileFoundry HIR: the reference every fast
implementation in ``runtime_model.py`` is measured against.

The published model (``GraniteMoeHybridForCausalLM``) is a 40-layer hybrid: 35
Mamba-2 layers and 5 full-attention layers in the order ``config.layer_types``
states, each followed by the same mixture-of-experts block plus a dense shared
MLP. Three things about it are unusual enough to be worth naming here, because
each is a place a fixture copied from another model would be quietly wrong:

* **The attention carries no position at all.** ``position_embedding_type`` is
  ``"nope"``, so ``GraniteMoeHybridModel`` builds no rotary embedding and the
  attention sees ``position_embeddings=None``. There is no RoPE in this file
  because there is none in the model; position information lives entirely in
  the Mamba layers' recurrence.
* **Four published scalar multipliers.** ``embedding_multiplier``,
  ``residual_multiplier``, ``attention_multiplier`` and ``logits_scaling`` are
  real parts of the arithmetic, not defaults left at one. Dropping any of them
  still produces plausible text.
* **Weights keep the checkpoint's own ``(out, in)`` orientation.** Every
  ``ConstTensor`` here is declared the way ``nn.Linear`` stores it, and the
  matmuls transpose explicitly. That costs the reference a transpose it does
  not need; it buys the runtime twin a matvec whose reduction axis is
  contiguous, which is the whole difference between a fast decode and a slow
  one, and it makes the alias table a rename rather than a rewrite.

The KV cache is the fixed-capacity ``cache_update`` form rather than
caller-managed concat: the shapes then stay static across steps, which is what
lets the runtime twin capture one decode step as a CUDA graph and replay it.
The visible window is an additive mask the step is handed, so a kernel never
has to reason about how much of the capacity is live.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)

from tilefoundry import DType, func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: F401 -- tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings
from tilefoundry.evaluator import to_torch_dtype
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


def published(path: Path | None = None) -> GraniteMoeHybridConfig:
    """The checkpoint's own configuration, read by the class HF uses.

    The file sits beside this module, so a copy of this directory carries its
    own dimensions and needs nothing importable around it.
    """
    path = Path(__file__).parent / "config.json" if path is None else path
    return GraniteMoeHybridConfig(**json.loads(path.read_text(encoding="utf-8")))


config = published()

#: The published dtype as the DSL spells it. The checkpoint stores its weights
#: at this precision, so it is what a kernel reading them consumes.
_DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[
    str(config.dtype).removeprefix("torch.")
]

# One token per step, for the prompt and for the continuation alike.
S = 1

_H = config.hidden_size
_EPS = config.rms_norm_eps

# --- full attention -------------------------------------------------------
_HQ = config.num_attention_heads
_HKV = config.num_key_value_heads
#: `head_dim` is unpublished for this checkpoint, so HF derives it; deriving it
#: is therefore what the model says, not a shortcut around a published field.
_HD = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
_G = _HQ // _HKV
#: Not `head_dim ** -0.5`: `attention_multiplier` is published (1/128 here,
#: which happens to be 1/head_dim rather than its square root).
_ATT_SCALE = config.attention_multiplier

#: KV capacity, fixed so a decode step has one static shape. `cur_pos` and the
#: additive mask say how much of it is live. Prompt plus continuation must fit.
MAX_CTX = int(os.environ.get("GRANITE_MAX_CTX", "4096"))
_CAP = MAX_CTX

# --- mamba-2 mixer --------------------------------------------------------
_NH = config.mamba_n_heads
_PD = config.mamba_d_head
_NS = config.mamba_d_state
_NG = config.mamba_n_groups
#: `mamba_expand * hidden_size`, which is also `n_heads * d_head`.
_EXP = config.mamba_expand * config.hidden_size
#: The depthwise convolution's channel count: the gated stream, then B, then C.
_CONVD = _EXP + 2 * _NG * _NS
_KRN = config.mamba_d_conv
#: How many earlier positions the causal convolution needs. The kernel spans
#: `d_conv` positions ending at the one being decoded, so the state handed in
#: is the `d_conv - 1` before it.
_WIN = _KRN - 1
#: `in_proj`'s fan-out: the gate, the convolved stream, and one dt per head.
_PROJ = _EXP + _CONVD + _NH
#: Heads sharing one B/C group. One group here, so every head shares one pair.
_HPG = _NH // _NG

# --- mixture of experts ---------------------------------------------------
_E = config.num_local_experts
_K = config.num_experts_per_tok
_I = config.intermediate_size
_IS = config.shared_intermediate_size

# --- published scalar multipliers ----------------------------------------
_EMB_MULT = config.embedding_multiplier
_RES_MULT = config.residual_multiplier
_LOGIT_SCALE = config.logits_scaling

_V = config.vocab_size

#: _DT as torch spells it -- the caches below are at the kernels' own dtype.
_TORCH_DT = to_torch_dtype(DType.from_name(_DT))
#: Masked-out attention positions. Finite, so a zeroed cache row scores
#: `0 * q + MASKED` rather than `inf - inf`, and bf16 holds it exactly enough
#: that `exp` of the difference is a clean zero.
_MASKED = -1.0e30


@module(entry="mamba_mixer")
class GraniteMamba:
    """`input_layernorm` + `GraniteMoeHybridMambaLayer` at one token per step.

    Split into four functions along the boundaries the arithmetic itself has:
    the projection, the depthwise convolution, the selective-state recurrence,
    and the gated output norm. Each is a thing a kernel can be judged on by
    itself.
    """

    @func
    def in_projection(
        hidden: Tensor[(1, S, _H), _DT],
        gamma_in: ConstTensor[(_H,), _DT],
        w_in: ConstTensor[(_PROJ, _H), _DT],
    ) -> Tensor[(1, _PROJ), _DT]:
        # The layer's pre-norm fused onto `in_proj`. One projection, three
        # consumers: the output gate, the convolved q/B/C stream, and dt.
        hidden_norm = tf.rms_norm(hidden, gamma_in, eps=_EPS)
        flat = tf.reshape(hidden_norm, new_shape=(S, _H))
        return tf.matmul(flat, tf.transpose(w_in, perm=(1, 0)))

    @func
    def conv_step(
        conv_state: Tensor[(1, _CONVD, _WIN), _DT],
        entry: Tensor[(1, _CONVD, S), _DT],
        conv_w: ConstTensor[(_CONVD, _KRN), _DT],
        conv_b: ConstTensor[(_CONVD,), _DT],
    ) -> Tensor[(1, _CONVD), _DT]:
        # The depthwise causal convolution at one token per step: the window
        # closes on this token, so the whole convolution is one multiply
        # against the kernel and one reduction over it. Channels do not mix --
        # that is what depthwise means here, and why no matmul appears.
        window = tf.concat(conv_state, entry, axis=2)
        weighted = window * tf.reshape(conv_w, new_shape=(1, _CONVD, _KRN))
        summed = tf.reduce(weighted, axes=(-1,), keepdim=False, kind="sum")
        return tf.silu(summed + tf.reshape(conv_b, new_shape=(1, _CONVD)))

    @func
    def ssm_step(
        ssm_state: Tensor[(1, _NH, _PD, _NS), "f32"],
        x: Tensor[(1, _NH, _PD), _DT],
        b_vec: Tensor[(1, _NG, _NS), _DT],
        c_vec: Tensor[(1, _NG, _NS), _DT],
        dt_raw: Tensor[(1, _NH), _DT],
        a_log: ConstTensor[(_NH,), _DT],
        dt_bias: ConstTensor[(_NH,), _DT],
        d_skip: ConstTensor[(_NH,), _DT],
    ):
        # One token of the selective state-space recurrence. Returns the
        # read-out and the state after this token; the state is an output
        # because a rank-one update has no smaller increment to hand back.
        #
        # `time_step_limit` is (0, inf) for this checkpoint, so the clamp HF
        # applies after the softplus is the identity and is not written here.
        dt = tf.softplus(dt_raw + tf.reshape(dt_bias, new_shape=(1, _NH)))
        # A is negative by construction, so exp(dt * A) is a decay in (0, 1):
        # the state cannot grow except through this token's own rank-one add.
        a = -tf.exp(tf.cast(a_log, dtype="f32"))
        decay = tf.exp(tf.cast(dt, dtype="f32") * tf.reshape(a, new_shape=(1, _NH)))

        # B and C are per group; with one group every head reads the same pair.
        dt_e = tf.reshape(dt, new_shape=(1, _NH, 1, 1))
        b_e = tf.reshape(
            tf.repeat_interleave(b_vec, repeats=_HPG, axis=1),
            new_shape=(1, _NH, 1, _NS),
        )
        c_e = tf.reshape(
            tf.repeat_interleave(c_vec, repeats=_HPG, axis=1),
            new_shape=(1, _NH, 1, _NS),
        )
        # dB then dBx, in that order: HF forms the discretised B first and
        # only then scales it by the token, and at bf16 the two orders differ.
        d_b = dt_e * b_e
        d_bx = d_b * tf.reshape(x, new_shape=(1, _NH, _PD, 1))

        updated = ssm_state * tf.reshape(decay, new_shape=(1, _NH, 1, 1)) + tf.cast(
            d_bx, dtype="f32"
        )
        # The read-out is a batched matvec over the state: HF rounds the state
        # to the activation dtype first and accumulates the products in f32,
        # which is what the double cast spells.
        read = tf.reduce(
            tf.cast(tf.cast(updated, dtype=_DT), dtype="f32") * tf.cast(c_e, dtype="f32"),
            axes=(-1,),
            keepdim=False,
            kind="sum",
        )
        y = tf.cast(read, dtype=_DT) + x * tf.reshape(d_skip, new_shape=(1, _NH, 1))
        return y, updated

    @func
    def gated_out(
        y: Tensor[(1, _NH, _PD), _DT],
        gate: Tensor[(1, _EXP), _DT],
        gamma_ssm: ConstTensor[(_EXP,), _DT],
        w_out: ConstTensor[(_H, _EXP), _DT],
    ) -> Tensor[(1, S, _H), _DT]:
        # `GraniteMoeHybridRMSNormGated` then `out_proj`. The gate multiplies
        # *before* the normalisation, so it changes the norm rather than only
        # scaling its result -- norm-before-gate would be a different model.
        flat = tf.reshape(y, new_shape=(1, _EXP))
        gated = tf.cast(flat, dtype="f32") * tf.silu(tf.cast(gate, dtype="f32"))
        normed = tf.cast(tf.rms_norm(gated, gamma_ssm, eps=_EPS), dtype=_DT)
        return tf.reshape(
            tf.matmul(normed, tf.transpose(w_out, perm=(1, 0))), new_shape=(1, S, _H)
        )

    @func
    def mamba_mixer(
        hidden: Tensor[(1, S, _H), _DT],
        gamma_in: ConstTensor[(_H,), _DT],
        w_in: ConstTensor[(_PROJ, _H), _DT],
        conv_w: ConstTensor[(_CONVD, _KRN), _DT],
        conv_b: ConstTensor[(_CONVD,), _DT],
        a_log: ConstTensor[(_NH,), _DT],
        dt_bias: ConstTensor[(_NH,), _DT],
        d_skip: ConstTensor[(_NH,), _DT],
        conv_state: Tensor[(1, _CONVD, _WIN), _DT],
        ssm_state: Tensor[(1, _NH, _PD, _NS), "f32"],
        gamma_ssm: ConstTensor[(_EXP,), _DT],
        w_out: ConstTensor[(_H, _EXP), _DT],
    ):
        # The whole mixer, no residual (the layer owns the residual add).
        # Returns the output, this step's own convolution column, and the
        # updated recurrent state -- the two halves the caller advances.
        proj = in_projection(hidden, gamma_in, w_in)
        gate = proj[:, :_EXP]
        entry = tf.reshape(proj[:, _EXP : _EXP + _CONVD], new_shape=(1, _CONVD, S))
        dt_raw = proj[:, _EXP + _CONVD : _PROJ]

        mixed = conv_step(conv_state, entry, conv_w, conv_b)
        x = tf.reshape(mixed[:, :_EXP], new_shape=(1, _NH, _PD))
        b_vec = tf.reshape(mixed[:, _EXP : _EXP + _NG * _NS], new_shape=(1, _NG, _NS))
        c_vec = tf.reshape(
            mixed[:, _EXP + _NG * _NS : _CONVD], new_shape=(1, _NG, _NS)
        )

        y, updated = ssm_step(ssm_state, x, b_vec, c_vec, dt_raw, a_log, dt_bias, d_skip)
        return gated_out(y, gate, gamma_ssm, w_out), entry, updated


@module(entry="full_attention")
class GraniteAttention:
    """`input_layernorm` + `GraniteMoeHybridAttention` at one token per step.

    No rotary anything: this checkpoint's `position_embedding_type` is
    ``"nope"``, so the query and key carry no position.
    """

    @func
    def qkv(
        hidden: Tensor[(1, S, _H), _DT],
        gamma_in: ConstTensor[(_H,), _DT],
        w_q: ConstTensor[(_HQ * _HD, _H), _DT],
        w_k: ConstTensor[(_HKV * _HD, _H), _DT],
        w_v: ConstTensor[(_HKV * _HD, _H), _DT],
    ):
        # The layer's pre-norm fused onto the three projections.
        hidden_norm = tf.reshape(
            tf.rms_norm(hidden, gamma_in, eps=_EPS), new_shape=(S, _H)
        )
        q = tf.reshape(
            tf.matmul(hidden_norm, tf.transpose(w_q, perm=(1, 0))),
            new_shape=(1, S, _HQ, _HD),
        )
        k = tf.reshape(
            tf.matmul(hidden_norm, tf.transpose(w_k, perm=(1, 0))),
            new_shape=(1, S, _HKV, _HD),
        )
        v = tf.reshape(
            tf.matmul(hidden_norm, tf.transpose(w_v, perm=(1, 0))),
            new_shape=(1, S, _HKV, _HD),
        )
        return q, k, v

    @func
    def attend(
        q: Tensor[(1, S, _HQ, _HD), _DT],
        k_cache: Tensor[(1, _CAP, _HKV, _HD), _DT],
        v_cache: Tensor[(1, _CAP, _HKV, _HD), _DT],
        cur_pos: Tensor[(1,), "i32"],
        mask: Tensor[(1, 1, _CAP), _DT],
        w_o: ConstTensor[(_H, _HQ * _HD), _DT],
    ) -> Tensor[(1, S, _H), _DT]:
        # Scores over the whole capacity; `mask` is what says which of it is
        # live. Softmax in f32 and back, as `eager_attention_forward` does.
        #
        # `cur_pos` says the same thing `mask` does, one number instead of a
        # row of them. The reference reads the row because a shaped tensor is
        # what HIR can express; an implementation is free to read the number
        # and stop the loop there instead, which is why both are declared.
        k_ctx = tf.repeat_interleave(
            tf.transpose(k_cache, perm=(0, 2, 1, 3)), repeats=_G, axis=1
        )
        v_ctx = tf.repeat_interleave(
            tf.transpose(v_cache, perm=(0, 2, 1, 3)), repeats=_G, axis=1
        )
        q_e = tf.reshape(q, new_shape=(1, _HQ, 1, _HD))
        scores = tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=False, kind="sum")
        scaled = tf.cast(scores, dtype="f32") * tf.full_like(
            tf.cast(scores, dtype="f32"), value=_ATT_SCALE
        ) + tf.cast(mask, dtype="f32")
        probs = tf.cast(tf.softmax(scaled, axis=-1), dtype=_DT)
        attn = tf.reduce(
            tf.reshape(probs, new_shape=(1, _HQ, _CAP, 1)) * v_ctx,
            axes=(-2,),
            keepdim=False,
            kind="sum",
        )
        flat = tf.reshape(attn, new_shape=(S, _HQ * _HD))
        return tf.reshape(
            tf.matmul(flat, tf.transpose(w_o, perm=(1, 0))), new_shape=(1, S, _H)
        )

    @func
    def full_attention(
        hidden: Tensor[(1, S, _H), _DT],
        gamma_in: ConstTensor[(_H,), _DT],
        w_q: ConstTensor[(_HQ * _HD, _H), _DT],
        w_k: ConstTensor[(_HKV * _HD, _H), _DT],
        w_v: ConstTensor[(_HKV * _HD, _H), _DT],
        k_cache: Tensor[(1, _CAP, _HKV, _HD), _DT],
        v_cache: Tensor[(1, _CAP, _HKV, _HD), _DT],
        cur_pos: Tensor[(1,), "i32"],
        mask: Tensor[(1, 1, _CAP), _DT],
        w_o: ConstTensor[(_H, _HQ * _HD), _DT],
    ):
        # No residual (the layer owns it). The two caches come back updated in
        # place of this step's entry, so the caller carries one pair forward
        # rather than appending to a tensor that changes shape.
        q, k, v = qkv(hidden, gamma_in, w_q, w_k, w_v)
        one = tf.full_like(cur_pos, value=1)
        k_next = tf.cache_update(k_cache, cur_pos, one, k)
        v_next = tf.cache_update(v_cache, cur_pos, one, v)
        return attend(q, k_next, v_next, cur_pos, mask, w_o), k_next, v_next


@module(entry="routing")
class GraniteRouter:
    """The block's expert selection, as a Module of its own so it loads and
    runs by itself. Its output is an index, so a router that picked a
    different ten would be a different model even if every weight matched."""

    @func
    def routing(
        tokens: Tensor[(S, _H), _DT],
        # Only ConstTensor parameters are bound by Module.load.
        w_router: ConstTensor[(_E, _H), _DT],
    ):
        # HF `GraniteMoeHybridTopKRouter`: logits in f32, the top k, then a
        # softmax over *those k alone* -- not a softmax over every expert
        # followed by a renormalisation, which is a different distribution.
        logits = tf.cast(tf.matmul(tokens, tf.transpose(w_router, perm=(1, 0))), dtype="f32")
        top_vals, indices = tf.topk(logits, k=_K, axis=-1)
        return tf.cast(tf.softmax(top_vals, axis=-1), dtype=_DT), indices


@module(entry="experts")
class GraniteMoE:
    """`post_attention_layernorm`, the sparse block, and the dense shared MLP
    that runs beside it. Everything heavy in the second half of a layer."""

    router = GraniteRouter

    @func
    def post_norm(
        hidden: Tensor[(1, S, _H), _DT],
        gamma_post: ConstTensor[(_H,), _DT],
    ) -> Tensor[(S, _H), _DT]:
        # Fused here rather than in the layer, and its own function because
        # the router reads its output.
        return tf.reshape(tf.rms_norm(hidden, gamma_post, eps=_EPS), new_shape=(S, _H))

    @func
    def routed_experts(
        tokens: Tensor[(S, _H), _DT],
        weights: Tensor[(S, _K), _DT],
        indices: Tensor[(S, _K), "i64"],
        w_gate: ConstTensor[(_E, _I, _H), _DT],
        w_up: ConstTensor[(_E, _I, _H), _DT],
        w_down: ConstTensor[(_E, _H, _I), _DT],
    ) -> Tensor[(S, _H), _DT]:
        # The gathers are the point: `indices` is a runtime value, so the
        # three expert tensors are indexed by it rather than sliced at a known
        # offset. Each token then runs `num_experts_per_tok` independent
        # SwiGLU experts and their outputs are mixed by the routing weights.
        gate_w = tf.gather(w_gate, indices, axis=0)
        up_w = tf.gather(w_up, indices, axis=0)
        down_w = tf.gather(w_down, indices, axis=0)
        token_col = tf.reshape(tokens, new_shape=(S, 1, _H, 1))
        gate = tf.reshape(tf.matmul(gate_w, token_col), new_shape=(S, _K, _I))
        up = tf.reshape(tf.matmul(up_w, token_col), new_shape=(S, _K, _I))
        inner = tf.silu(gate) * up
        down = tf.reshape(
            tf.matmul(down_w, tf.reshape(inner, new_shape=(S, _K, _I, 1))),
            new_shape=(S, _K, _H),
        )
        weighted = down * tf.reshape(weights, new_shape=(S, _K, 1))
        return tf.reduce(weighted, axes=(1,), keepdim=False, kind="sum")

    @func
    def shared_mlp(
        tokens: Tensor[(S, _H), _DT],
        w_shared_in: ConstTensor[(2 * _IS, _H), _DT],
        w_shared_out: ConstTensor[(_H, _IS), _DT],
    ) -> Tensor[(S, _H), _DT]:
        # `GraniteMoeHybridMLP`: one fused projection chunked in two, the
        # first half activated and the second used as the multiplicand. Every
        # token goes through it, whichever experts it also picked.
        both = tf.matmul(tokens, tf.transpose(w_shared_in, perm=(1, 0)))
        gate = both[:, :_IS]
        up = both[:, _IS : 2 * _IS]
        return tf.matmul(tf.silu(gate) * up, tf.transpose(w_shared_out, perm=(1, 0)))

    @func
    def experts(
        tokens: Tensor[(S, _H), _DT],
        weights: Tensor[(S, _K), _DT],
        indices: Tensor[(S, _K), "i64"],
        w_gate: ConstTensor[(_E, _I, _H), _DT],
        w_up: ConstTensor[(_E, _I, _H), _DT],
        w_down: ConstTensor[(_E, _H, _I), _DT],
        w_shared_in: ConstTensor[(2 * _IS, _H), _DT],
        w_shared_out: ConstTensor[(_H, _IS), _DT],
    ) -> Tensor[(1, S, _H), _DT]:
        # The sparse block and the dense one are summed, not chosen between.
        # No residual -- the layer owns the residual add.
        routed = routed_experts(tokens, weights, indices, w_gate, w_up, w_down)
        shared = shared_mlp(tokens, w_shared_in, w_shared_out)
        return tf.reshape(routed + shared, new_shape=(1, S, _H))

    def forward(self, hidden):
        """One decode step of the block: post-norm, route, then the experts."""
        tokens = self.post_norm(hidden)
        weights, indices = self.router.routing(tokens)
        return self.experts(tokens, weights, indices)


def _layer_forward(self, hidden, mixer_args):
    """One decode step: mixer + scaled residual, then MoE + scaled residual.

    Mirrors ``GraniteMoeHybridDecoderLayer.forward``. The two pre-norms are not
    here because each block fuses its own, so each fused kernel lines up with
    one Hugging Face pre-norm-then-block composition.

    What comes back is the layer output and whatever state the mixer produced,
    passed through untouched for the caller to advance.
    """
    mixed, *state = self.mixer(hidden, *mixer_args)
    attended = self.residual_add(hidden, mixed)
    expert_out = self.moe(attended)
    return self.residual_add(attended, expert_out), tuple(state)


@module(entry="residual_add")
class GraniteMambaDecoderLayer:
    mixer = GraniteMamba.renamed("mixer")
    moe = GraniteMoE.renamed("moe")

    @func
    def residual_add(
        a: Tensor[(1, S, _H), _DT],
        b: Tensor[(1, S, _H), _DT],
    ) -> Tensor[(1, S, _H), _DT]:
        # `residual + hidden_states * residual_multiplier`: the multiplier is
        # published and is not 1, so it is part of the arithmetic.
        return a + b * tf.full_like(b, value=_RES_MULT)

    forward = _layer_forward


@module(entry="residual_add")
class GraniteAttentionDecoderLayer:
    mixer = GraniteAttention.renamed("mixer")
    moe = GraniteMoE.renamed("moe")

    @func
    def residual_add(
        a: Tensor[(1, S, _H), _DT],
        b: Tensor[(1, S, _H), _DT],
    ) -> Tensor[(1, S, _H), _DT]:
        return a + b * tf.full_like(b, value=_RES_MULT)

    forward = _layer_forward


#: Which layer class each published `layer_types` entry names. The model states
#: this, not its tests: it is the same fact `config.layer_types` is written in.
LAYER_TYPE = {
    "linear_attention": GraniteMambaDecoderLayer,
    "full_attention": GraniteAttentionDecoderLayer,
}

#: The parameters a mixer declares for its own state, whichever kind it is. The
#: root splices a layer's cache in at the first of them.
_CACHE_PARAMS = frozenset({"k_cache", "v_cache", "conv_state", "ssm_state"})


def _with_cache(mixer, mixer_args, cache):
    """*mixer_args* with *cache* spliced in where *mixer* declares its state.

    The position is counted over the parameters a step is handed, since a
    loading fills the weights by name, and read from the Module a loading
    stands over so that one rule answers for both.
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

    A Mamba layer's convolution window slides by the one column the step
    produced, while its recurrent state is replaced whole -- a rank-one update
    has no smaller increment. An attention layer's caches were written in
    place by `cache_update`, so its step already handed back the next state.
    """
    import torch  # noqa: PLC0415

    if kind == "linear_attention":
        window, _matrix = state
        column, updated = fresh
        return torch.cat([window, column], dim=2)[:, :, -_WIN:], updated
    return fresh


@lru_cache(maxsize=None)
def _mask_row(device, dtype):
    """A reusable `[0, 0, ..., MASKED, ...]` row builder's constant halves."""
    import torch  # noqa: PLC0415

    return (
        torch.zeros(_CAP, device=device, dtype=dtype),
        torch.full((_CAP,), _MASKED, device=device, dtype=dtype),
    )


def attention_mask(step, device, dtype=None):
    """The additive mask a step at position *step* attends under.

    Positions `0 .. step` are live -- the step's own key and value are written
    into the cache before the scores are taken -- and everything past it is
    masked out.
    """
    import torch  # noqa: PLC0415

    dtype = _TORCH_DT if dtype is None else dtype
    live, dead = _mask_row(device, dtype)
    row = torch.where(
        torch.arange(_CAP, device=device) <= step, live, dead
    )
    return row.reshape(1, 1, _CAP)


@module(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 132), Topology("thread", 256)))
class Granite4_0_H_Small:
    """The layer stack in `config.layer_types` order, and the step around it --
    embedding, the walk, the closing norm, the head. Each layer is an
    independent copy, so an analysis of one annotates only it."""


    # The published layer-type cycle determines each layer Module.
    layers = tuple(
        LAYER_TYPE[kind].renamed(f"layer{index}")
        for index, kind in enumerate(config.layer_types)
    )

    @func
    def embed(
        table: ConstTensor[(_V, _H), _DT],
        token_ids: Tensor[(1,), "i64"],
    ) -> Tensor[(1, S, _H), _DT]:
        # HF `GraniteMoeHybridModel.embed_tokens`, times the published
        # `embedding_multiplier` -- which is 12 here, not 1.
        row = tf.gather(table, token_ids, axis=0)
        return tf.reshape(
            row * tf.full_like(row, value=_EMB_MULT), new_shape=(1, S, _H)
        )

    @func
    def final_rms_norm(
        hidden: Tensor[(1, S, _H), _DT],
        gamma_final: ConstTensor[(_H,), _DT],
    ) -> Tensor[(1, S, _H), _DT]:
        # HF `GraniteMoeHybridModel.norm`, applied once after the last layer.
        return tf.rms_norm(hidden, gamma_final, eps=_EPS)

    @func
    def lm_head(
        hidden: Tensor[(1, S, _H), _DT],
        table: ConstTensor[(_V, _H), _DT],
    ) -> Tensor[(1, _V), _DT]:
        # `tie_word_embeddings` is true, so the head *is* the embedding table
        # -- declared under the same name so one reading binds one tensor and
        # the 822 MB is read once. Divided by the published `logits_scaling`.
        logits = tf.matmul(
            tf.reshape(hidden, new_shape=(1, _H)), tf.transpose(table, perm=(1, 0))
        )
        return logits / tf.full_like(logits, value=_LOGIT_SCALE)

    def decode_hidden(self, hidden, layer_args, caches):
        """One decode step through every layer, then the final norm.

        *layer_args* is one layer's mixer arguments per layer, carrying no
        state; *caches* is each layer's own state, spliced into its mixer
        call. What comes back is the normed hidden state and each layer's
        fresh state.
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

        The fresh per-layer state comes out beside the logits; carrying
        *caches* forward with it is the caller's step, through `append_cache`.
        """
        hidden = self.embed(token_ids)
        normed, states = self.decode_hidden(hidden, layer_args, caches)
        return self.lm_head(normed), states

    def init_caches(self, device=None):
        """The per-layer state container, one entry per layer.

        A Mamba layer's two halves are genuinely zero at the start: Hugging
        Face left-pads the convolution window when the context is shorter than
        it, and an uninitialised recurrent state is the zero matrix. An
        attention layer gets its whole capacity, zeroed so that a position the
        mask excludes still scores a finite number.
        """
        import torch  # noqa: PLC0415

        device = torch.accelerator.current_accelerator() if device is None else device
        entries = []
        for kind in config.layer_types:
            if kind == "linear_attention":
                entries.append((
                    torch.zeros(1, _CONVD, _WIN, dtype=_TORCH_DT, device=device),
                    torch.zeros(1, _NH, _PD, _NS, dtype=torch.float32, device=device),
                ))
            else:
                empty = torch.zeros(1, _CAP, _HKV, _HD, dtype=_TORCH_DT, device=device)
                entries.append((empty, empty.clone()))
        return tuple(entries)

    def append_cache(self, caches, fresh):
        """Every layer's state advanced by the step it just took."""
        return tuple(
            advance_state(kind, cache, new)
            for kind, cache, new in zip(config.layer_types, caches, fresh)
        )

    def prepare_inputs_for_generation(self, input_ids, step, caches, device=None):
        """The token and each layer's non-state activations for one step."""
        import torch  # noqa: PLC0415

        device = torch.accelerator.current_accelerator() if device is None else device
        token_ids = input_ids[step].reshape(1).to(device=device, dtype=torch.int64)
        cur_pos = torch.full((1,), step, device=device, dtype=torch.int32)
        mask = attention_mask(step, device)
        layer_args = tuple(
            () if kind == "linear_attention" else (cur_pos, mask)
            for kind in config.layer_types
        )
        return token_ids, layer_args, caches
