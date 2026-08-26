"""Qwen3.5-35B-A3B's published submodules and both of its decoder layer types.

The two layer types differ only in which token mixer they hold. Each is its own
Module rather than one Module with a branch, because they are different kernels:
a branch would give analysis and scheduling one domain for two behaviours.
They share the walk, which composes Modules and so cannot be a ``@func``.

Provenance
----------
The `@func` bodies below are the shipped source of `qwen3_5_35b_a3b`
(`tilefoundry models qwen3_5_35b_a3b --source`), verbatim. Three things are
added, and nothing is removed:

1. `config` comes from the sibling `config.py` instead of `tests.models...`,
   which does not ship (see that file's docstring).
2. The class bodies live inside `build(config)`. Parser §2.7 blesses exactly
   this for "a model asked about more than one structural configuration"; it is
   what lets a 4-layer stack and the published 40-layer one be the same source.
   The module-level names below are `build(REAL)`, so a selector still reads
   `model.py:Qwen3_5Decoder.layer0.mixer`.
3. **Per-weight converters.** The shipped source registers exactly one
   (`lm_head.w_head`), which is enough at L1 -- "per-op against Hugging Face
   with random weights" never binds a real checkpoint. Step one is finished
   only "when the authored Module agrees with the published implementation on
   real weights", and real weights need the rest of them: every projection is
   stored transposed relative to what the matmuls here want, and every
   `Qwen3_5MoeRMSNorm` gamma is stored as a *delta from one*
   (`output * (1.0 + self.weight)` in `modeling_qwen3_5_moe.py`) while
   `tf.rms_norm` is `x * weight` flat (hir §rmsnorm). Binding the raw tensor
   would be wrong by that `1 +` on 163 of the 285 gammas, in a way every shape
   check accepts. `linear_attn.norm.weight` is the exception: it feeds
   `Qwen3_5MoeRMSNormGated`, which is flat, and its stored values sit near 1
   rather than near 0.

`hf_alias(config)` at the bottom is the {canonical: raw} table those converters
read through -- runtime §1.5's `alias`, with `Absolute` where a child consumes a
tensor its parent owns (`input_layernorm.weight` is the layer's, not the
mixer's).
"""
from __future__ import annotations

import math
import os
import sys

# `tilefoundry check` / `analyze` load a SOURCE with `runpy.run_path`, which does
# not put the file's own directory on `sys.path`. Without this, the sibling
# `config` import below resolves under `python -c "import model"` and not under
# the CLI. See FINDINGS Q2.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import REAL  # noqa: E402 -- must follow the sys.path bootstrap
from tilefoundry import DType, func, module
from tilefoundry.target import CudaTarget
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: F401 -- tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings
from tilefoundry.evaluator import to_torch_dtype
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.runtime import Absolute


def build(config):
    """The whole model at one structural configuration.

    Returns the namespace the module level below re-exports: the two mixers, the
    router, the MoE block, the two layer kinds, the decoder, and the two helpers
    the decoder's orchestration methods need.
    """
    # One token per step. No other extent here is dynamic either: this mixer's
    # state is fixed-size, so the module carries no DimVar.
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

    # The delta rule's query scale, and the epsilon its L2 normalisation uses.
    # Both are architecture constants rather than runtime values -- they are
    # fixed by ``linear_key_head_dim``, not chosen per step -- so they are folded
    # in here instead of taking up parameters a caller would have to get right.
    _QSCALE = 1.0 / math.sqrt(config.gdn_head_k_dim)
    _L2_EPS = 1e-6

    # Prior-cache length. The caller appends this step's returned K/V entry.
    C = DimVar("ctx_len", 0, config.max_ctx)

    _HQ = config.n_q_heads
    _HKV = config.n_kv_heads
    # Published dimensions; do not derive them from the other fields.
    _D = config.head_dim
    _ROT = config.rotary_dim
    _PASS = config.pass_dim
    _G = config.gqa_group

    # One row per position a step may be decoded at: `pos_ids` is the
    # prior-cache length, which stops one below ``max_ctx``.
    # ``max_position_embeddings`` is 262144 and a cache that size is 67 MB of
    # zeros nothing reads.
    _ROPE_ROWS = config.max_ctx

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
            # closes on this token, so the whole convolution is one multiply
            # against the kernel and one reduction over it. Channels do not mix
            # -- that is what depthwise means here, and it is why no matmul
            # appears.
            window = tf.concat([conv_state, entry], axis=2)
            weighted = window * tf.reshape(conv_w, new_shape=(1, _CONV, _KERNEL))
            summed = tf.reduce(weighted, axes=(-1,), keepdim=False, kind="sum")
            return tf.silu(summed)

        @func
        def l2_normalise(
            x: Tensor[(1, S, _HV, _DK), config.dt],
        ) -> Tensor[(1, S, _HV, _DK), config.dt]:
            # Per-head L2 normalisation, matching the linear-attention library's
            # own (`l2norm` in the Hugging Face module): rsqrt of the *sum* of
            # squares plus eps, not of the mean, so it is not an RMSNorm with a
            # unit scale.
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
            # One token of the gated delta rule. Returns the read-out and the
            # updated state, in that order; the state is an output because a
            # rank-one update has no smaller increment to hand back.
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
            w_in_qkv: ConstTensor[(1, _H, _CONV), config.dt_w],
            w_in_z: ConstTensor[(1, _H, _VAL), config.dt_w],
            w_in_b: ConstTensor[(1, _H, _HV), config.dt_w],
            w_in_a: ConstTensor[(1, _H, _HV), config.dt_w],
            conv_w: ConstTensor[(_CONV, _KERNEL), config.dt],
            a_log: ConstTensor[(_HV,), config.dt],
            dt_bias: ConstTensor[(_HV,), config.dt],
            conv_state: Tensor[(1, _CONV, _WINDOW), config.dt],
            recurrent_state: Tensor[(1, _HV, _DK, _DV), config.dt],
            gamma_gdn: ConstTensor[(_DV,), config.dt],
            w_out: ConstTensor[(1, _VAL, _H), config.dt_w],
        ):
            # Fused input_layernorm + `Qwen3_5MoeGatedDeltaNet`, no residual (the
            # layer owns the residual add). Returns the output, this step's own
            # convolution column, and the updated recurrent state.
            hidden_norm = tf.rms_norm(hidden, gamma_in)

            entry = tf.transpose(tf.cast(tf.matmul(tf.cast(hidden_norm, dtype=config.dt_w), w_in_qkv), dtype=config.dt), perm=(0, 2, 1))
            mixed = conv_step(conv_state, entry, conv_w)

            q_flat = mixed[:, :_KEY]
            k_flat = mixed[:, _KEY : 2 * _KEY]
            v_flat = mixed[:, 2 * _KEY : _CONV]

            # Every value head reads the key head it shares; the projection
            # produces one key head per group, and the delta rule runs per value
            # head.
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

            beta = tf.sigmoid(tf.cast(tf.matmul(tf.cast(hidden_norm, dtype=config.dt_w), w_in_b), dtype=config.dt))
            # g is negative by construction, so exp(g) is a decay in (0, 1): the
            # state cannot grow without a token asking for it through the
            # rank-one update.
            g = -tf.exp(a_log) * tf.softplus(tf.cast(tf.matmul(tf.cast(hidden_norm, dtype=config.dt_w), w_in_a), dtype=config.dt) + dt_bias)

            read, updated = delta_step(recurrent_state, q, k, v, g, beta)

            # The gated output norm: normalise per value head, scale, then gate
            # by a projection of the layer input through silu.
            z = tf.reshape(tf.cast(tf.matmul(tf.cast(hidden_norm, dtype=config.dt_w), w_in_z), dtype=config.dt), new_shape=(1, _HV, _DV))
            normed = tf.rms_norm(read, gamma_gdn)
            gated = normed * tf.silu(z)
            out = tf.cast(tf.matmul(tf.cast(tf.reshape(gated, new_shape=(1, S, _VAL)), dtype=config.dt_w), w_out), dtype=config.dt)
            return out, entry, updated

        # ---- raw checkpoint -> declared weight ---------------------------

        @linear_attention.converter("gamma_in")
        def _(
            input_layernorm_weight: ConstTensor[(_H,), config.dt],
        ) -> Tensor[(_H,), config.dt]:
            # `Qwen3_5MoeRMSNorm` is `x * (1 + w)`; `tf.rms_norm` is `x * w`.
            return input_layernorm_weight + tf.full_like(
                input_layernorm_weight, value=1.0
            )

        @linear_attention.converter("w_in_qkv")
        def _(
            in_proj_qkv_weight: ConstTensor[(_CONV, _H), config.dt_w],
        ) -> Tensor[(1, _H, _CONV), config.dt_w]:
            # `nn.Linear` stores (out, in); every matmul above is (in, out).
            return tf.reshape(
                tf.transpose(in_proj_qkv_weight, perm=(1, 0)), new_shape=(1, _H, _CONV)
            )

        @linear_attention.converter("w_in_z")
        def _(
            in_proj_z_weight: ConstTensor[(_VAL, _H), config.dt_w],
        ) -> Tensor[(1, _H, _VAL), config.dt_w]:
            return tf.reshape(
                tf.transpose(in_proj_z_weight, perm=(1, 0)), new_shape=(1, _H, _VAL)
            )

        @linear_attention.converter("w_in_b")
        def _(
            in_proj_b_weight: ConstTensor[(_HV, _H), config.dt_w],
        ) -> Tensor[(1, _H, _HV), config.dt_w]:
            return tf.reshape(
                tf.transpose(in_proj_b_weight, perm=(1, 0)), new_shape=(1, _H, _HV)
            )

        @linear_attention.converter("w_in_a")
        def _(
            in_proj_a_weight: ConstTensor[(_HV, _H), config.dt_w],
        ) -> Tensor[(1, _H, _HV), config.dt_w]:
            return tf.reshape(
                tf.transpose(in_proj_a_weight, perm=(1, 0)), new_shape=(1, _H, _HV)
            )

        @linear_attention.converter("conv_w")
        def _(
            conv1d_weight: ConstTensor[(_CONV, 1, _KERNEL), config.dt],
        ) -> Tensor[(_CONV, _KERNEL), config.dt]:
            # A depthwise `nn.Conv1d` keeps a length-1 in-channel axis.
            return tf.reshape(conv1d_weight, new_shape=(_CONV, _KERNEL))

        @linear_attention.converter("w_out")
        def _(
            out_proj_weight: ConstTensor[(_H, _VAL), config.dt_w],
        ) -> Tensor[(1, _VAL, _H), config.dt_w]:
            return tf.reshape(
                tf.transpose(out_proj_weight, perm=(1, 0)), new_shape=(1, _VAL, _H)
            )

        # `a_log`, `dt_bias` and `gamma_gdn` need no converter: the first two are
        # stored exactly as declared, and `gamma_gdn` feeds the *gated* RMSNorm,
        # which has no `1 +`.

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
            # whole of its input's last axis, so the split is what makes a
            # partial factor expressible at all rather than an optional
            # rearrangement.
            rot = x[:, :, :, :_ROT]
            tail = x[:, :, :, _ROT:_D]
            turned, _ = tf.rope(rot, rot, cos_cache, sin_cache, pos_ids)
            return tf.concat([turned, tail], axis=-1)

        @func
        def partial_rope_kv(
            x: Tensor[(1, S, _HKV, _D), config.dt],
            cos_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
            sin_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
            pos_ids: Tensor[(S,), "i32"],
        ) -> Tensor[(1, S, _HKV, _D), config.dt]:
            # The same rotation over the key's head count. Its own Function
            # because a Function's parameter shapes are fixed and GQA's two head
            # counts differ.
            rot = x[:, :, :, :_ROT]
            tail = x[:, :, :, _ROT:_D]
            turned, _ = tf.rope(rot, rot, cos_cache, sin_cache, pos_ids)
            return tf.concat([turned, tail], axis=-1)

        @func
        def full_attention(
            hidden: Tensor[(1, S, _H), config.dt],
            gamma_in: ConstTensor[(_H,), config.dt],
            w_qg: ConstTensor[(1, _H, _HQ * _D * 2), config.dt_w],
            w_k: ConstTensor[(1, _H, _HKV * _D), config.dt_w],
            w_v: ConstTensor[(1, _H, _HKV * _D), config.dt_w],
            gamma_q: ConstTensor[(_D,), config.dt],
            gamma_k: ConstTensor[(_D,), config.dt],
            cos_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
            sin_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
            pos_ids: Tensor[(S,), "i32"],
            k_cache: Tensor[(1, C, _HKV, _D), config.dt],
            v_cache: Tensor[(1, C, _HKV, _D), config.dt],
            scale: Tensor[(1, 1, 1, 1), config.dt],
            w_o: ConstTensor[(1, _HQ * _D, _H), config.dt_w],
        ):
            # Fused input_layernorm + `Qwen3_5MoeAttention`, no residual (the
            # layer owns the residual add). Returns the attention output together
            # with this token's key and value, which are what the caller appends
            # to the cache.
            hidden_norm = tf.rms_norm(hidden, gamma_in)

            # One projection, two halves: the query and the output gate. The
            # split is over the last axis of the [heads, 2 * head_dim] view, so
            # gate entry j of head h sits beside query entry j of the same head,
            # not in a second contiguous block of the flat projection.
            qg = tf.reshape(
                tf.cast(tf.matmul(tf.cast(hidden_norm, dtype=config.dt_w), w_qg), dtype=config.dt), new_shape=(1, S, _HQ, 2 * _D)
            )
            q = qg[:, :, :, :_D]
            gate = qg[:, :, :, _D : 2 * _D]

            q_rope = partial_rope(
                tf.rms_norm(q, gamma_q), cos_cache, sin_cache, pos_ids
            )
            k_rope = partial_rope_kv(
                tf.rms_norm(
                    tf.reshape(
                        tf.cast(tf.matmul(tf.cast(hidden_norm, dtype=config.dt_w), w_k), dtype=config.dt), new_shape=(1, S, _HKV, _D)
                    ),
                    gamma_k,
                ),
                cos_cache, sin_cache, pos_ids,
            )
            v = tf.reshape(tf.cast(tf.matmul(tf.cast(hidden_norm, dtype=config.dt_w), w_v), dtype=config.dt), new_shape=(1, S, _HKV, _D))

            # Every query head sees its group's key/value head, for the cache and
            # for the new token alike.
            q_s = q_rope * scale
            k_ctx = tf.reshape(
                tf.transpose(
                    tf.repeat_interleave(k_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)
                ),
                new_shape=(1, 1, _HQ, C, _D),
            )
            v_ctx = tf.reshape(
                tf.transpose(
                    tf.repeat_interleave(v_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)
                ),
                new_shape=(1, 1, _HQ, C, _D),
            )
            k_new = tf.repeat_interleave(k_rope, repeats=_G, axis=2)
            v_new = tf.repeat_interleave(v, repeats=_G, axis=2)

            # Two score groups: one over the cache, one over the token itself.
            q_e = tf.reshape(q_s, new_shape=(1, S, _HQ, 1, _D))
            score_ctx = tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=True, kind="sum")
            score_new = tf.reduce(q_s * k_new, axes=(-1,), keepdim=True, kind="sum")

            # Log-sum-exp merge of the two groups' partials against their joint
            # max.
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

            # The output gate, then o_proj. Head-major flattening on both sides,
            # so gate entry (h, j) meets attention entry (h, j).
            gated = tf.reshape(attn, new_shape=(1, S, _HQ * _D)) * tf.sigmoid(
                tf.reshape(gate, new_shape=(1, S, _HQ * _D))
            )
            return tf.cast(tf.matmul(tf.cast(gated, dtype=config.dt_w), w_o), dtype=config.dt), k_rope, v

        # ---- raw checkpoint -> declared weight ---------------------------

        @full_attention.converter("gamma_in")
        def _(
            input_layernorm_weight: ConstTensor[(_H,), config.dt],
        ) -> Tensor[(_H,), config.dt]:
            return input_layernorm_weight + tf.full_like(
                input_layernorm_weight, value=1.0
            )

        @full_attention.converter("w_qg")
        def _(
            q_proj_weight: ConstTensor[(_HQ * _D * 2, _H), config.dt_w],
        ) -> Tensor[(1, _H, _HQ * _D * 2), config.dt_w]:
            # `attn_output_gate` doubles q_proj's width, already head-major with
            # the gate beside the query inside each head -- which is the layout
            # `full_attention` slices. So this is only the (out, in) transpose.
            return tf.reshape(
                tf.transpose(q_proj_weight, perm=(1, 0)),
                new_shape=(1, _H, _HQ * _D * 2),
            )

        @full_attention.converter("w_k")
        def _(
            k_proj_weight: ConstTensor[(_HKV * _D, _H), config.dt_w],
        ) -> Tensor[(1, _H, _HKV * _D), config.dt_w]:
            return tf.reshape(
                tf.transpose(k_proj_weight, perm=(1, 0)),
                new_shape=(1, _H, _HKV * _D),
            )

        @full_attention.converter("w_v")
        def _(
            v_proj_weight: ConstTensor[(_HKV * _D, _H), config.dt_w],
        ) -> Tensor[(1, _H, _HKV * _D), config.dt_w]:
            return tf.reshape(
                tf.transpose(v_proj_weight, perm=(1, 0)),
                new_shape=(1, _H, _HKV * _D),
            )

        @full_attention.converter("gamma_q")
        def _(
            q_norm_weight: ConstTensor[(_D,), config.dt],
        ) -> Tensor[(_D,), config.dt]:
            return q_norm_weight + tf.full_like(q_norm_weight, value=1.0)

        @full_attention.converter("gamma_k")
        def _(
            k_norm_weight: ConstTensor[(_D,), config.dt],
        ) -> Tensor[(_D,), config.dt]:
            return k_norm_weight + tf.full_like(k_norm_weight, value=1.0)

        @full_attention.converter("w_o")
        def _(
            o_proj_weight: ConstTensor[(_H, _HQ * _D), config.dt_w],
        ) -> Tensor[(1, _HQ * _D, _H), config.dt_w]:
            return tf.reshape(
                tf.transpose(o_proj_weight, perm=(1, 0)),
                new_shape=(1, _HQ * _D, _H),
            )

    @module(entry="routing")
    class Qwen3_5Router:
        """The block's expert selection, as a Module of its own so it loads and
        runs by itself. Its output is an index, so a router that picked a
        different eight would be a different model even if every weight
        matched."""

        @func
        def routing(
            tokens: Tensor[(S, _H), config.dt],
            # Only ConstTensor parameters are bound by Module.load.
            w_router: ConstTensor[(_H, _E), config.dt_w],
        ):
            # HF `Qwen3_5MoeTopKRouter`: softmax over every expert in f32, then
            # the top k, then renormalise.
            logits = tf.cast(tf.matmul(tf.cast(tokens, dtype=config.dt_w), w_router), dtype="f32")
            probs = tf.softmax(logits, axis=-1)
            top_vals, indices = tf.topk(probs, k=_K, axis=-1)
            denom = tf.reduce(top_vals, axes=(-1,), keepdim=True, kind="sum")
            return tf.cast(top_vals / denom, dtype=config.dt), indices

        @routing.converter("w_router")
        def _(
            gate_weight: ConstTensor[(_E, _H), config.dt_w],
        ) -> Tensor[(_H, _E), config.dt_w]:
            return tf.transpose(gate_weight, perm=(1, 0))

    @module(entry="experts")
    class Qwen3_5MoE:
        router = Qwen3_5Router

        @func
        def post_norm(
            hidden: Tensor[(1, S, _H), config.dt],
            gamma_post: ConstTensor[(_H,), config.dt],
        ) -> Tensor[(S, _H), config.dt]:
            # HF `post_attention_layernorm`, fused here rather than in the layer,
            # and its own function because the router reads its output.
            return tf.reshape(tf.rms_norm(hidden, gamma_post), new_shape=(S, _H))

        @func
        def routed_experts(
            tokens: Tensor[(S, _H), config.dt],
            weights: Tensor[(S, _K), config.dt],
            indices: Tensor[(S, _K), "i64"],
            w_gate: ConstTensor[(_E, _I, _H), config.dt_w],
            w_up: ConstTensor[(_E, _I, _H), config.dt_w],
            w_down: ConstTensor[(_E, _H, _I), config.dt_w],
        ) -> Tensor[(S, _H), config.dt]:
            # The gathers are the point: `indices` is a runtime value, so the
            # three expert tensors are indexed by it rather than sliced at a
            # known offset. Each token then runs `top_k` independent SwiGLU
            # experts, batched over the (token, slot) pair, and their outputs are
            # mixed by the routing weights.
            flat_indices = tf.reshape(indices, new_shape=(S * _K,))
            gate_w = tf.reshape(
                tf.index_select(w_gate, flat_indices, dim=0),
                new_shape=(S, _K, _I, _H),
            )
            up_w = tf.reshape(
                tf.index_select(w_up, flat_indices, dim=0),
                new_shape=(S, _K, _I, _H),
            )
            down_w = tf.reshape(
                tf.index_select(w_down, flat_indices, dim=0),
                new_shape=(S, _K, _H, _I),
            )
            token_col = tf.reshape(tokens, new_shape=(S, 1, _H, 1))
            gate = tf.reshape(tf.cast(tf.matmul(gate_w, tf.cast(token_col, dtype=config.dt_w)), dtype=config.dt), new_shape=(S, _K, _I))
            up = tf.reshape(tf.cast(tf.matmul(up_w, tf.cast(token_col, dtype=config.dt_w)), dtype=config.dt), new_shape=(S, _K, _I))
            hidden = tf.silu(gate) * up
            down = tf.reshape(
                tf.cast(tf.matmul(down_w, tf.cast(tf.reshape(hidden, new_shape=(S, _K, _I, 1)), dtype=config.dt_w)), dtype=config.dt),
                new_shape=(S, _K, _H),
            )
            weighted = down * tf.reshape(weights, new_shape=(S, _K, 1))
            return tf.reduce(weighted, axes=(1,), keepdim=False, kind="sum")

        @func
        def shared_expert(
            tokens: Tensor[(S, _H), config.dt],
            w_shared_gate: ConstTensor[(_H, _IS), config.dt_w],
            w_shared_up: ConstTensor[(_H, _IS), config.dt_w],
            w_shared_down: ConstTensor[(_IS, _H), config.dt_w],
            w_shared_scale: ConstTensor[(_H, 1), config.dt_w],
        ) -> Tensor[(S, _H), config.dt]:
            # A dense SwiGLU every token goes through, scaled by the token's own
            # scalar gate. The gate is a projection to width one through a
            # sigmoid, so it is between 0 and 1 per token and cannot change sign.
            gate = tf.cast(tf.matmul(tf.cast(tokens, dtype=config.dt_w), w_shared_gate), dtype=config.dt)
            up = tf.cast(tf.matmul(tf.cast(tokens, dtype=config.dt_w), w_shared_up), dtype=config.dt)
            dense = tf.cast(tf.matmul(tf.cast(tf.silu(gate) * up, dtype=config.dt_w), w_shared_down), dtype=config.dt)
            scale = tf.sigmoid(tf.cast(tf.matmul(tf.cast(tokens, dtype=config.dt_w), w_shared_scale), dtype=config.dt))
            return dense * scale

        @func
        def experts(
            tokens: Tensor[(S, _H), config.dt],
            weights: Tensor[(S, _K), config.dt],
            indices: Tensor[(S, _K), "i64"],
            w_gate: ConstTensor[(_E, _I, _H), config.dt_w],
            w_up: ConstTensor[(_E, _I, _H), config.dt_w],
            w_down: ConstTensor[(_E, _H, _I), config.dt_w],
            w_shared_gate: ConstTensor[(_H, _IS), config.dt_w],
            w_shared_up: ConstTensor[(_H, _IS), config.dt_w],
            w_shared_down: ConstTensor[(_IS, _H), config.dt_w],
            w_shared_scale: ConstTensor[(_H, 1), config.dt_w],
        ) -> Tensor[(1, S, _H), config.dt]:
            # `Qwen3_5MoeSparseMoeBlock` once the selection is made, and
            # everything in the block that is heavy: the routed experts, the
            # dense shared one, and their mix. No residual -- the layer owns the
            # residual add.
            routed = routed_experts(tokens, weights, indices, w_gate, w_up, w_down)
            shared = shared_expert(
                tokens, w_shared_gate, w_shared_up, w_shared_down, w_shared_scale
            )
            return tf.reshape(routed + shared, new_shape=(1, S, _H))

        def forward(self, hidden):
            """One decode step of the block: post-norm, route, then the
            experts."""
            tokens = self.post_norm(hidden)
            weights, indices = self.router.routing(tokens)
            return self.experts(tokens, weights, indices)

        # ---- raw checkpoint -> declared weight ---------------------------

        @post_norm.converter("gamma_post")
        def _(
            post_attention_layernorm_weight: ConstTensor[(_H,), config.dt],
        ) -> Tensor[(_H,), config.dt]:
            return post_attention_layernorm_weight + tf.full_like(
                post_attention_layernorm_weight, value=1.0
            )

        @routed_experts.converter("w_gate")
        def _(
            gate_up_proj: ConstTensor[(_E, 2 * _I, _H), config.dt_w],
        ) -> Tensor[(_E, _I, _H), config.dt_w]:
            # HF fuses the two halves of the SwiGLU into one (E, 2I, H) tensor
            # and splits with `.chunk(2, dim=-1)` on the *output* of the linear,
            # i.e. contiguous halves of the 2I axis -- gate first.
            return gate_up_proj[:, :_I, :]

        @routed_experts.converter("w_up")
        def _(
            gate_up_proj: ConstTensor[(_E, 2 * _I, _H), config.dt_w],
        ) -> Tensor[(_E, _I, _H), config.dt_w]:
            return gate_up_proj[:, _I : 2 * _I, :]

        @shared_expert.converter("w_shared_gate")
        def _(
            shared_gate_proj_weight: ConstTensor[(_IS, _H), config.dt_w],
        ) -> Tensor[(_H, _IS), config.dt_w]:
            return tf.transpose(shared_gate_proj_weight, perm=(1, 0))

        @shared_expert.converter("w_shared_up")
        def _(
            shared_up_proj_weight: ConstTensor[(_IS, _H), config.dt_w],
        ) -> Tensor[(_H, _IS), config.dt_w]:
            return tf.transpose(shared_up_proj_weight, perm=(1, 0))

        @shared_expert.converter("w_shared_down")
        def _(
            shared_down_proj_weight: ConstTensor[(_H, _IS), config.dt_w],
        ) -> Tensor[(_IS, _H), config.dt_w]:
            return tf.transpose(shared_down_proj_weight, perm=(1, 0))

        @shared_expert.converter("w_shared_scale")
        def _(
            shared_expert_gate_weight: ConstTensor[(1, _H), config.dt_w],
        ) -> Tensor[(_H, 1), config.dt_w]:
            return tf.transpose(shared_expert_gate_weight, perm=(1, 0))

        # `w_down` needs no converter: HF stores it (E, H, I), which is what the
        # matmul above wants.

    def _layer_forward(self, hidden, mixer_args):
        """One decode step: mixer + residual, then MoE + residual.

        Mirrors ``Qwen3_5MoeDecoderLayer.forward``. The two pre-norms are not
        here because each block fuses its own -- the mixer fuses
        ``input_layernorm`` and the MoE block fuses
        ``post_attention_layernorm``, so each fused kernel lines up with one
        Hugging Face pre-norm-then-block composition.

        *mixer_args* is what the mixer is handed after the hidden state. The MoE
        block is handed the mixed state and nothing else: every weight it reads
        is one it holds.

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

    #: Which layer class each published `layer_types` entry names. The model
    #: states this, not its tests: it is the same fact `config.layer_types` is
    #: written in.
    LAYER_TYPE = {
        "full_attention": Qwen3_5FullAttnLayer,
        "linear_attention": Qwen3_5LinearAttnLayer,
    }

    #: config.dt as torch spells it -- the state below is at the kernels' own
    #: dtype.
    _TORCH_DT = to_torch_dtype(DType.from_name(config.dt))

    #: The parameters a mixer declares for its own state, whichever kind it is.
    #: The root splices a layer's cache in at the first of them.
    _CACHE_PARAMS = frozenset(
        {"k_cache", "v_cache", "conv_state", "recurrent_state"}
    )

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
        # `next`, not `min`: `from tilefoundry.dsl.tf import *` binds `min` to
        # the op.
        at = next(index for index, name in enumerate(names) if name in _CACHE_PARAMS)
        return (*mixer_args[:at], *cache, *mixer_args[at:])

    def advance_state(kind, state, fresh):
        """A layer of *kind*'s next state, from what its mixer returned.

        The recurrent matrix is replaced whole -- a rank-one update has no
        smaller increment -- while the convolution window slides by the one
        column the step produced. Key and value are appended.
        """
        import torch  # noqa: PLC0415

        if kind == "linear_attention":
            window, _matrix = state
            column, updated = fresh
            return torch.cat([window, column], dim=2)[:, :, -_WINDOW:], updated
        return tuple(torch.cat([old, new], dim=1) for old, new in zip(state, fresh))

    # The only Target declaration in the tree. Target §6: "Only the outermost
    # `Module` of a tree declares a `Target`; every Module below it inherits that
    # one declaration and MUST NOT declare its own." Declaring it is what makes
    # `tilefoundry analyze` / `schedule` / `inspect capabilities` answer at all --
    # they refuse to resolve an undeclared Target to a default, because
    # "measuring or scheduling against a device the author never declared is a
    # silent wrong answer". The constructed H200 SXM Target states 132 SMs,
    # 4.8 TB/s of HBM, and 989.5 TFLOP/s dense bf16 from the installed document.
    @module(target=CudaTarget("nvidia.h200_sxm"))
    class Qwen3_5Decoder:
        """The layer stack in `config.layer_types` order, and the step around it
        -- embedding, the walk, the closing norm, the head. Each layer is an
        independent copy, so an analysis of one annotates only it."""

        # The published layer-type cycle determines each layer Module.
        layers = tuple(
            LAYER_TYPE[kind].renamed(f"layer{index}")
            for index, kind in enumerate(config.layer_types)
        )

        @func
        def embed(
            table: ConstTensor[(config.vocab, config.hidden), config.dt_w],
            token_ids: Tensor[(1,), "i64"],
        ) -> Tensor[(1, S, config.hidden), config.dt]:
            # HF `Qwen3_5MoeModel.embed_tokens`: the decoded token's own row.
            return tf.reshape(
                tf.index_select(table, token_ids, dim=0),
                new_shape=(1, S, config.hidden),
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
            w_head: ConstTensor[(config.hidden, config.vocab), config.dt_w],
        ) -> Tensor[(1, config.vocab), config.dt]:
            # HF `Qwen3_5MoeForCausalLM.lm_head`, over the one token being
            # decoded.
            return tf.cast(
                tf.matmul(
                    tf.cast(tf.reshape(hidden, new_shape=(1, config.hidden)), dtype=config.dt_w),
                    w_head,
                ),
                dtype=config.dt,
            )

        @lm_head.converter("w_head")
        def _(
            head_weight_raw: ConstTensor[(config.vocab, config.hidden), config.dt_w],
        ) -> Tensor[(config.hidden, config.vocab), config.dt_w]:
            # HF stores the head as (vocab, hidden); the matmul above wants it
            # the other way. Tied models alias this input to the embedding table.
            return tf.transpose(head_weight_raw, perm=(1, 0))

        @final_rms_norm.converter("gamma_final")
        def _(
            final_norm_weight: ConstTensor[(config.hidden,), config.dt],
        ) -> Tensor[(config.hidden,), config.dt]:
            return final_norm_weight + tf.full_like(final_norm_weight, value=1.0)

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
                hidden, state = layer(
                    hidden, _with_cache(layer.mixer, mixer_args, cache)
                )
                states.append(state)
            return self.final_rms_norm(hidden), tuple(states)

        def forward(self, token_ids, layer_args, caches):
            """One decode step of the whole model: token ids in, logits out.

            The fresh per-layer state comes out beside the logits; growing
            *caches* with it is the caller's step, through `append_cache`.
            """
            hidden = self.embed(token_ids)
            normed, states = self.decode_hidden(hidden, layer_args, caches)
            return self.lm_head(normed), states

        def init_caches(self, device="cuda"):
            """The per-layer state container, one entry per layer.

            A linear-attention layer's two halves are genuinely zero at the
            start: Hugging Face left-pads the convolution window when the
            context is shorter than it, and `initial_state=None` is the zero
            recurrent matrix. An attention layer gets a container of no
            positions, which `ctx_len` admits: the first step of a sequence
            attends the one position it brings itself.
            """
            import torch  # noqa: PLC0415

            entries = []
            for kind in config.layer_types:
                if kind == "linear_attention":
                    entries.append((
                        torch.zeros(
                            1, _CONV, _WINDOW, dtype=_TORCH_DT, device=device
                        ),
                        torch.zeros(
                            1, _HV, _DK, _DV, dtype=_TORCH_DT, device=device
                        ),
                    ))
                else:
                    empty = torch.zeros(
                        1, 0, _HKV, _D, dtype=_TORCH_DT, device=device
                    )
                    entries.append((empty, empty))
            return tuple(entries)

        def append_cache(self, caches, fresh):
            """Every layer's state advanced by the step it just took: a kernel
            hands back its own token's entry, and joining it on is the
            caller's."""
            return tuple(
                advance_state(kind, cache, new)
                for kind, cache, new in zip(config.layer_types, caches, fresh)
            )

    return {
        "config": config,
        "Qwen3_5LinearAttention": Qwen3_5LinearAttention,
        "Qwen3_5FullAttention": Qwen3_5FullAttention,
        "Qwen3_5Router": Qwen3_5Router,
        "Qwen3_5MoE": Qwen3_5MoE,
        "Qwen3_5FullAttnLayer": Qwen3_5FullAttnLayer,
        "Qwen3_5LinearAttnLayer": Qwen3_5LinearAttnLayer,
        "Qwen3_5Decoder": Qwen3_5Decoder,
        "LAYER_TYPE": LAYER_TYPE,
        "advance_state": advance_state,
    }


# ---------------------------------------------------------------------------
# The published configuration, as module-level names a CLI selector can address.
# ---------------------------------------------------------------------------

_REAL = build(REAL)

config = _REAL["config"]
Qwen3_5LinearAttention = _REAL["Qwen3_5LinearAttention"]
Qwen3_5FullAttention = _REAL["Qwen3_5FullAttention"]
Qwen3_5Router = _REAL["Qwen3_5Router"]
Qwen3_5MoE = _REAL["Qwen3_5MoE"]
Qwen3_5FullAttnLayer = _REAL["Qwen3_5FullAttnLayer"]
Qwen3_5LinearAttnLayer = _REAL["Qwen3_5LinearAttnLayer"]
Qwen3_5Decoder = _REAL["Qwen3_5Decoder"]
LAYER_TYPE = _REAL["LAYER_TYPE"]
advance_state = _REAL["advance_state"]


# ---------------------------------------------------------------------------
# {canonical: raw} for the published checkpoint.
# ---------------------------------------------------------------------------

#: The raw names each mixer / block / decoder weight (or converter parameter) is
#: stored under, relative to the Hugging Face text decoder.
_MIXER_RAW = {
    "linear_attention": {
        "in_proj_qkv_weight": "linear_attn.in_proj_qkv.weight",
        "in_proj_z_weight": "linear_attn.in_proj_z.weight",
        "in_proj_b_weight": "linear_attn.in_proj_b.weight",
        "in_proj_a_weight": "linear_attn.in_proj_a.weight",
        "conv1d_weight": "linear_attn.conv1d.weight",
        "a_log": "linear_attn.A_log",
        "dt_bias": "linear_attn.dt_bias",
        "gamma_gdn": "linear_attn.norm.weight",
        "out_proj_weight": "linear_attn.out_proj.weight",
        "input_layernorm_weight": "input_layernorm.weight",
    },
    "full_attention": {
        "q_proj_weight": "self_attn.q_proj.weight",
        "k_proj_weight": "self_attn.k_proj.weight",
        "v_proj_weight": "self_attn.v_proj.weight",
        "q_norm_weight": "self_attn.q_norm.weight",
        "k_norm_weight": "self_attn.k_norm.weight",
        "o_proj_weight": "self_attn.o_proj.weight",
        "input_layernorm_weight": "input_layernorm.weight",
    },
}

_MOE_RAW = {
    "post_attention_layernorm_weight": "post_attention_layernorm.weight",
    "gate_up_proj": "mlp.experts.gate_up_proj",
    "w_down": "mlp.experts.down_proj",
    "shared_gate_proj_weight": "mlp.shared_expert.gate_proj.weight",
    "shared_up_proj_weight": "mlp.shared_expert.up_proj.weight",
    "shared_down_proj_weight": "mlp.shared_expert.down_proj.weight",
    "shared_expert_gate_weight": "mlp.shared_expert_gate.weight",
}

_ROUTER_RAW = {"gate_weight": "mlp.gate.weight"}


def hf_alias(cfg=REAL, *, text_prefix="model.language_model", head="lm_head.weight"):
    """The alias table `Qwen3_5Decoder.prepare` / `.load` reads the published
    checkpoint through.

    Every entry is `Absolute`, and every key is fully path-qualified. Runtime
    §1.5 says aliasing "only ever reaches downward" -- a mixer cannot address
    `input_layernorm.weight`, which its parent layer owns -- so the escape is
    needed for real weights, not only for tidiness. Fully qualifying the rest
    costs one dict entry per weight and buys the property that no lookup depends
    on which prefix a `subtree` happened to accumulate.
    """
    alias: dict[str, object] = {
        "table": Absolute(f"{text_prefix}.embed_tokens.weight"),
        "final_norm_weight": Absolute(f"{text_prefix}.norm.weight"),
        "head_weight_raw": Absolute(head),
    }
    for index, kind in enumerate(cfg.layer_types):
        layer = f"{text_prefix}.layers.{index}"
        for canonical, raw in _MIXER_RAW[kind].items():
            alias[f"layer{index}.mixer.{canonical}"] = Absolute(f"{layer}.{raw}")
        for canonical, raw in _MOE_RAW.items():
            alias[f"layer{index}.moe.{canonical}"] = Absolute(f"{layer}.{raw}")
        for canonical, raw in _ROUTER_RAW.items():
            alias[f"layer{index}.moe.router.{canonical}"] = Absolute(f"{layer}.{raw}")
    return alias


def leaf_alias(kind_or_node, layer_index, cfg=REAL, *, text_prefix="model.language_model"):
    """`hf_alias` scoped to one leaf, for loading a submodule on its own.

    `tilefoundry check model.py:Qwen3_5LinearAttention --ckpt ...` stands over
    the bare mixer rather than the decoder, so its resource has no `layerN.mixer`
    prefix to qualify against.
    """
    layer = f"{text_prefix}.layers.{layer_index}"
    if kind_or_node in _MIXER_RAW:
        table = _MIXER_RAW[kind_or_node]
    elif kind_or_node == "moe":
        table = {**_MOE_RAW, **{f"router.{k}": v for k, v in _ROUTER_RAW.items()}}
    elif kind_or_node == "router":
        table = _ROUTER_RAW
    else:
        raise ValueError(f"no raw table for {kind_or_node!r}")
    return {
        canonical: Absolute(f"{layer}.{raw}") for canonical, raw in table.items()
    }


__all__ = [
    "LAYER_TYPE",
    "Qwen3_5Decoder",
    "Qwen3_5FullAttention",
    "Qwen3_5FullAttnLayer",
    "Qwen3_5LinearAttention",
    "Qwen3_5LinearAttnLayer",
    "Qwen3_5MoE",
    "Qwen3_5Router",
    "advance_state",
    "build",
    "config",
    "hf_alias",
    "leaf_alias",
]
