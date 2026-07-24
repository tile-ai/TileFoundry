"""Qwen3.5-35B-A3B GatedDeltaNet (linear-attention) component: 30 of the 40
decoder layers per config.json's ``layer_types`` (the other 10 use
``attention.py``'s full attention). Mirrors
``transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`` (transformers
5.12.1, cited below as M) decode step, M:369-558.

``gdn_mix``: input RMSNorm -> ``in_proj_qkv`` -> single-step grouped causal
conv1d (k=4, silu, M:220-236) advancing ``conv_state`` -> split q(2048)/
k(2048)/v(4096), q/k 16 heads ``repeat_interleave``'d to 32 (M:500/518) ->
gates (f32, M:514-516): ``beta = sigmoid(in_proj_b(x))``, ``g = -exp(A_log)
* softplus(in_proj_a(x) + dt_bias)`` -> f32 recurrence over ``rec_state``
(q/k l2norm'd first, q scaled by ``GDN_SCALE``, M:325-368): ``S *= e^g``;
``kv_mem = sum_k(S * k)``; ``delta = (v - kv_mem) * beta``; ``S += k (x)
delta``; ``out = sum_k(S * q)`` -> RMSNormGated (M:185-200):
``norm(out) * w * silu(z)`` -> ``out_proj`` -> residual.

``gdn_convert``: the sole RAW-checkpoint -> CANONICAL (kernel-native layout)
weight conversion entry point for ``gdn_mix`` -- pure repack/cast, no
numeric change. Unlike the other RMSNorms in this model, RMSNormGated's
weight (M:187) is ones-initialized and multiplied as-is (no ``1 + w``).
"""
from __future__ import annotations

from tests.models.qwen3_5_35b_a3b.config import HIDDEN
from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.module import Module

GDN_KEY_DIM = 2048       # 16 heads x 128
GDN_VALUE_DIM = 4096     # 32 heads x 128
GDN_CONV_DIM = 8192      # key x2 + value
GDN_HEADS = 32
GDN_HDIM = 128
GDN_CONV_K = 4
GDN_EPS = 1e-6           # l2norm (M:238) shares this eps with RMSNormGated
GDN_SCALE = GDN_HDIM ** -0.5
GDN_INV_HDIM = 1.0 / GDN_HDIM


@func
def gdn_mix(
    x: Tensor[(1, 1, HIDDEN), "bf16"],
    in_norm_raw: ConstTensor[(HIDDEN,), "f32"],              # RAW (not +1); (1+w) applied in-body
    w_qkv: ConstTensor[(GDN_CONV_DIM, HIDDEN), "bf16"],      # kernel-native [out,in] (see gdn_convert)
    w_z: ConstTensor[(GDN_VALUE_DIM, HIDDEN), "bf16"],
    w_b: ConstTensor[(GDN_HEADS, HIDDEN), "bf16"],
    w_a: ConstTensor[(GDN_HEADS, HIDDEN), "bf16"],
    conv_w: ConstTensor[(GDN_CONV_DIM, GDN_CONV_K), "bf16"],
    a_log: ConstTensor[(GDN_HEADS,), "f32"],
    dt_bias: ConstTensor[(GDN_HEADS,), "f32"],
    gated_norm_gamma: ConstTensor[(GDN_HDIM,), "bf16"],      # as-is weight: RMSNormGated (M:187) is ones-init, no (1+w)
    w_out: ConstTensor[(HIDDEN, GDN_VALUE_DIM), "bf16"],     # kernel-native [out,in]
    conv_state: Tensor[(1, GDN_CONV_DIM, GDN_CONV_K), "bf16"],
    rec_state: Tensor[(1, GDN_HEADS, GDN_HDIM, GDN_HDIM), "f32"],
):
    # Returns (y[1,1,2048], conv_state', rec_state'). Semantics: M:369-558
    # decode branch. CANONICAL weight layout: w_qkv/w_z/w_b/w_a/w_out are
    # nn.Linear-native [out,in], transposed in-body before matmul --
    # evaluator (this func) and the kernel override read the same
    # self.weights (sole conversion entry: gdn_convert).
    # ── input_layernorm (RAW weight, (1+w) applied in-body) ──────────────────
    h = tf.rms_norm(x, 1.0 + in_norm_raw)

    # ── in_proj_qkv -> single conv1d step (M:220-236: append state, keep the
    #    last GDN_CONV_K, depthwise conv = weighted sum along the kernel dim,
    #    silu) ──────────────────────────────────────────────────────────────
    mixed = tf.reshape(tf.matmul(h, tf.transpose(w_qkv, perm=(1, 0))), new_shape=(1, GDN_CONV_DIM, 1))
    tail = tf.slice(
        conv_state, begin=(0, 0, 1), end=(1, GDN_CONV_DIM, GDN_CONV_K), strides=(1, 1, 1)
    )
    conv_state_new = tf.concat(tail, mixed, axis=-1)          # (1,8192,4)
    conv_out = tf.reduce(
        conv_state_new * tf.reshape(conv_w, new_shape=(1, GDN_CONV_DIM, GDN_CONV_K)),
        axes=(-1,), keepdim=False, kind=ReduceKind.SUM,
    )                                                         # (1,8192)
    conv_out = conv_out * tf.sigmoid(conv_out)                # silu (M:233)

    # ── split q/k/v, 16 heads -> repeat_interleave -> 32 (M:500/518) ─────────
    q = tf.reshape(
        tf.slice(conv_out, begin=(0, 0), end=(1, GDN_KEY_DIM), strides=(1, 1)),
        new_shape=(1, GDN_HEADS // 2, GDN_HDIM),
    )
    k = tf.reshape(
        tf.slice(conv_out, begin=(0, GDN_KEY_DIM), end=(1, 2 * GDN_KEY_DIM), strides=(1, 1)),
        new_shape=(1, GDN_HEADS // 2, GDN_HDIM),
    )
    v = tf.reshape(
        tf.slice(conv_out, begin=(0, 2 * GDN_KEY_DIM), end=(1, GDN_CONV_DIM), strides=(1, 1)),
        new_shape=(1, GDN_HEADS, GDN_HDIM),
    )
    q = tf.repeat_interleave(q, repeats=2, axis=1)            # (1,32,128)
    k = tf.repeat_interleave(k, repeats=2, axis=1)

    # ── gates (M:514-516, f32): beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias)
    # beta: sigmoid in bf16 (M:513 projection dtype) before entering the f32 recurrence
    beta = tf.cast(
        tf.sigmoid(tf.reshape(tf.matmul(h, tf.transpose(w_b, perm=(1, 0))), new_shape=(1, GDN_HEADS))),
        dtype="f32",
    )
    a_prj = tf.cast(
        tf.reshape(tf.matmul(h, tf.transpose(w_a, perm=(1, 0))), new_shape=(1, GDN_HEADS)), dtype="f32"
    )
    g = tf.neg(tf.exp(a_log)) * tf.softplus(a_prj + dt_bias)  # (1,32) f32
    g_exp = tf.reshape(tf.exp(g), new_shape=(1, GDN_HEADS, 1, 1))

    # ── recurrence (M:325-368, f32; q/k l2norm first, q then scaled) ────────
    # l2norm done in bf16 (M:328-331, before .to(f32)), upcast to f32 after
    q_ss = tf.reduce(q * q, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    q = q * tf.rsqrt(q_ss + tf.full_like(q_ss, value=GDN_EPS))
    k_ss = tf.reduce(k * k, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    k = k * tf.rsqrt(k_ss + tf.full_like(k_ss, value=GDN_EPS))
    qf = tf.cast(q, dtype="f32") * tf.full_like(tf.cast(q, dtype="f32"), value=GDN_SCALE)
    kf = tf.cast(k, dtype="f32")
    vf = tf.cast(v, dtype="f32")

    s_decayed = rec_state * g_exp                             # (1,32,128k,128v)
    kv_mem = tf.reduce(
        s_decayed * tf.reshape(kf, new_shape=(1, GDN_HEADS, GDN_HDIM, 1)),
        axes=(2,), keepdim=False, kind=ReduceKind.SUM,
    )                                                         # (1,32,128v)
    delta = (vf - kv_mem) * tf.reshape(beta, new_shape=(1, GDN_HEADS, 1))
    rec_state_new = s_decayed + tf.reshape(kf, new_shape=(1, GDN_HEADS, GDN_HDIM, 1)) * tf.reshape(
        delta, new_shape=(1, GDN_HEADS, 1, GDN_HDIM)
    )
    out = tf.reduce(
        rec_state_new * tf.reshape(qf, new_shape=(1, GDN_HEADS, GDN_HDIM, 1)),
        axes=(2,), keepdim=False, kind=ReduceKind.SUM,
    )                                                         # (1,32,128) f32

    # ── RMSNormGated (M:185-200): recurrence output rounds to bf16 first
    #    (M:366 initial_dtype), norm upcasts to f32 -> rsqrt -> back to bf16
    #    x w (as-is) -> x silu(z.f32) -> bf16 ───────────────────────────────
    out_bf = tf.cast(out, dtype="bf16")
    of = tf.cast(out_bf, dtype="f32")
    o_ss = tf.reduce(of * of, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    o_ms = o_ss * tf.full_like(o_ss, value=GDN_INV_HDIM)
    o_n = tf.cast(of * tf.rsqrt(o_ms + tf.full_like(o_ms, value=GDN_EPS)), dtype="bf16")
    o_n = o_n * gated_norm_gamma                              # M:198 (weight as-is, not 1+w)
    z = tf.cast(
        tf.reshape(tf.matmul(h, tf.transpose(w_z, perm=(1, 0))), new_shape=(1, GDN_HEADS, GDN_HDIM)),
        dtype="f32",
    )
    o_g = tf.cast(o_n, dtype="f32") * (z * tf.sigmoid(z))     # M:199
    o_flat = tf.reshape(tf.cast(o_g, dtype="bf16"), new_shape=(1, 1, GDN_VALUE_DIM))

    # ── out_proj -> residual ──────────────────────────────────────────────
    y = tf.matmul(o_flat, tf.transpose(w_out, perm=(1, 0)))
    return x + y, conv_state_new, rec_state_new


@func
def gdn_convert(
    input_layernorm: ConstTensor[(HIDDEN,), "f32"],                     # RAW ckpt layernorm weight (not +1)
    in_proj_qkv: ConstTensor[(GDN_CONV_DIM, HIDDEN), "bf16"],           # nn.Linear-native [out,in]
    in_proj_z: ConstTensor[(GDN_VALUE_DIM, HIDDEN), "bf16"],
    in_proj_b: ConstTensor[(GDN_HEADS, HIDDEN), "bf16"],
    in_proj_a: ConstTensor[(GDN_HEADS, HIDDEN), "bf16"],
    conv1d_weight: ConstTensor[(GDN_CONV_DIM, 1, GDN_CONV_K), "bf16"],  # nn.Conv1d-native [C,1,K]
    a_log: ConstTensor[(GDN_HEADS,), "f32"],
    dt_bias: ConstTensor[(GDN_HEADS,), "f32"],
    norm_weight: ConstTensor[(GDN_HDIM,), "bf16"],                      # RMSNormGated weight, as-is
    out_proj: ConstTensor[(HIDDEN, GDN_VALUE_DIM), "bf16"],             # nn.Linear-native [out,in]
):
    # Returns (in_norm_raw, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias,
    # gated_norm_gamma, w_out): the CANONICAL weights for gdn_mix. Sole
    # conversion entry point; evaluator (via gdn_mix) and the kernel override
    # read this same product. Everything stays nn.Linear/nn.Conv1d-native
    # [out,in] (no transpose); norms stay RAW (not +1); the conv1d weight
    # squeezes out the groups dim. Pure repack/cast, no numeric change.
    in_norm_raw = tf.cast(input_layernorm, dtype="f32")
    w_qkv = tf.cast(in_proj_qkv, dtype="bf16")
    w_z = tf.cast(in_proj_z, dtype="bf16")
    w_b = tf.cast(in_proj_b, dtype="bf16")
    w_a = tf.cast(in_proj_a, dtype="bf16")
    conv_w = tf.reshape(tf.cast(conv1d_weight, dtype="bf16"), new_shape=(GDN_CONV_DIM, GDN_CONV_K))
    a_log_raw = tf.cast(a_log, dtype="f32")
    dt_bias_raw = tf.cast(dt_bias, dtype="f32")
    gated_norm_gamma = tf.cast(norm_weight, dtype="bf16")
    w_out = tf.cast(out_proj, dtype="bf16")
    return in_norm_raw, w_qkv, w_z, w_b, w_a, conv_w, a_log_raw, dt_bias_raw, gated_norm_gamma, w_out


gdn_module = Module(name="gdn", functions=(gdn_mix, gdn_convert), entry="gdn_mix")
