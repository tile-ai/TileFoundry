"""Qwen3.5-35B-A3B's ``linear_attention`` token mixer -- a Gated DeltaNet -- as
one tilefoundry IR Module, over a free ``config`` name; not importable on its
own, load it with ``tests.models.loader.load_model`` (see
``../linear_attention.py``).

Three layers in four are this one. It is not attention with a cheaper score: it
keeps no per-position history at all. What it carries between steps is

- a **recurrent matrix** per value head, ``[key_head_dim, value_head_dim]``, and
- a short **convolution window**, the ``linear_conv_kernel_dim - 1`` positions
  before this one, over the channels the projection produces.

Neither grows with the context, so this kernel carries no ``ctx_len`` and no
range at all: every extent in it is a literal. That is the real difference
between the two token mixers in this model, and it is why they are two
boundaries -- a fixture that shared one shape between them would have to
pretend one of them has a context length.

The contract is the same one the KV-cache layers keep, in the shape this mixer's
state takes: prior state in, read-only; this step's own contribution out; the
caller advances. For the recurrent matrix "this step's contribution" *is* the
whole updated matrix -- a rank-one update leaves nothing smaller to hand back --
so what comes out is the new matrix. For the convolution the step produces one
new column and the caller slides its window, exactly as it appends a key.

The step itself, following ``torch_recurrent_gated_delta_rule``:

- the projection's output goes through a depthwise causal convolution spanning
  ``linear_conv_kernel_dim`` positions, then silu. At one token per step that
  convolution is a weighted sum over the window -- there is no sliding to do --
  so it is written as one multiply and one reduction rather than as a conv op.
- query and key are L2-normalised per head; the query is then scaled by
  ``1/sqrt(key_head_dim)``.
- the recurrent matrix decays by ``exp(g)``, where ``g`` is negative:
  ``-exp(A_log) * softplus(a + dt_bias)``. It then takes a rank-one update
  toward the value, in the amount ``beta`` the token asks for --
  ``delta = (v - state @ k) * beta`` -- which is the delta rule the layer is
  named for, and the reason the state cannot be accumulated out of order.
- the output is read out of the *updated* state by the query, normalised, and
  gated by a separate projection ``z`` through silu.

``silu(x) = x * sigmoid(x)``; there is no standalone silu in the HIR op surface.
"""
from __future__ import annotations

import math

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 -- tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings

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


@module(entry="linear_attention")
class Qwen3_5LinearAttention:
    @func
    def conv_step(
        conv_state: Tensor[(1, _CONV, _WINDOW), config.dt],
        entry: Tensor[(1, _CONV, S), config.dt],
        conv_w: Tensor[(_CONV, _KERNEL), config.dt],
    ) -> Tensor[(1, _CONV), config.dt]:
        # The depthwise causal convolution at one token per step: the window
        # closes on this token, so the whole convolution is one multiply against
        # the kernel and one reduction over it. Channels do not mix -- that is
        # what depthwise means here, and it is why no matmul appears.
        window = tf.concat(conv_state, entry, axis=2)
        weighted = tf.mul(window, tf.reshape(conv_w, new_shape=(1, _CONV, _KERNEL)))
        summed = tf.reduce(weighted, axes=(-1,), keepdim=False, kind="sum")
        return tf.mul(summed, tf.sigmoid(summed))

    @func
    def l2_normalise(
        x: Tensor[(1, S, _HV, _DK), config.dt],
    ) -> Tensor[(1, S, _HV, _DK), config.dt]:
        # Per-head L2 normalisation, matching the linear-attention library's own
        # (`l2norm` in the Hugging Face module): rsqrt of the *sum* of squares
        # plus eps, not of the mean, so it is not an RMSNorm with a unit scale.
        square_sum = tf.reduce(tf.square(x), axes=(-1,), keepdim=True, kind="sum")
        return tf.mul(x, tf.rsqrt(tf.add(square_sum, tf.full_like(square_sum, value=_L2_EPS))))

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
        decayed = tf.mul(recurrent_state, tf.reshape(tf.exp(g), new_shape=(1, _HV, 1, 1)))
        k_col = tf.reshape(k, new_shape=(1, _HV, _DK, 1))
        recalled = tf.reduce(
            tf.mul(decayed, k_col), axes=(-2,), keepdim=False, kind="sum"
        )
        delta = tf.mul(
            tf.sub(tf.reshape(v, new_shape=(1, _HV, _DV)), recalled),
            tf.reshape(beta, new_shape=(1, _HV, 1)),
        )
        updated = tf.add(
            decayed, tf.mul(k_col, tf.reshape(delta, new_shape=(1, _HV, 1, _DV)))
        )
        q_scaled = tf.mul(q, tf.full_like(q, value=_QSCALE))
        read = tf.reduce(
            tf.mul(updated, tf.reshape(q_scaled, new_shape=(1, _HV, _DK, 1))),
            axes=(-2,), keepdim=False, kind="sum",
        )
        return read, updated

    @func
    def linear_attention(
        hidden: Tensor[(1, S, _H), config.dt],
        gamma_in: Tensor[(_H,), config.dt],
        w_in_qkv: Tensor[(1, _H, _CONV), config.dt],
        w_in_z: Tensor[(1, _H, _VAL), config.dt],
        w_in_b: Tensor[(1, _H, _HV), config.dt],
        w_in_a: Tensor[(1, _H, _HV), config.dt],
        conv_w: Tensor[(_CONV, _KERNEL), config.dt],
        a_log: Tensor[(_HV,), config.dt],
        dt_bias: Tensor[(_HV,), config.dt],
        conv_state: Tensor[(1, _CONV, _WINDOW), config.dt],
        recurrent_state: Tensor[(1, _HV, _DK, _DV), config.dt],
        gamma_gdn: Tensor[(_DV,), config.dt],
        w_out: Tensor[(1, _VAL, _H), config.dt],
    ):
        # Fused input_layernorm + `Qwen3_5MoeGatedDeltaNet`, no residual (the
        # layer owns the residual add). Returns the output, this step's own
        # convolution column, and the updated recurrent state.
        hidden_norm = tf.rms_norm(hidden, gamma_in)

        entry = tf.transpose(tf.matmul(hidden_norm, w_in_qkv), perm=(0, 2, 1))
        mixed = conv_step(conv_state, entry, conv_w)

        q_flat = tf.slice(mixed, begin=(0, 0), end=(1, _KEY), strides=(1, 1))
        k_flat = tf.slice(mixed, begin=(0, _KEY), end=(1, 2 * _KEY), strides=(1, 1))
        v_flat = tf.slice(mixed, begin=(0, 2 * _KEY), end=(1, _CONV), strides=(1, 1))

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
        g = tf.mul(
            tf.neg(tf.exp(a_log)),
            tf.softplus(tf.add(tf.matmul(hidden_norm, w_in_a), dt_bias)),
        )

        read, updated = delta_step(recurrent_state, q, k, v, g, beta)

        # The gated output norm: normalise per value head, scale, then gate by a
        # projection of the layer input through silu.
        z = tf.reshape(tf.matmul(hidden_norm, w_in_z), new_shape=(1, _HV, _DV))
        normed = tf.rms_norm(read, gamma_gdn)
        gated = tf.mul(normed, tf.mul(z, tf.sigmoid(z)))
        out = tf.matmul(tf.reshape(gated, new_shape=(1, S, _VAL)), w_out)
        return out, entry, updated
