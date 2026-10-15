"""Qwen3.8-27B text prefill, authored as HIR: the reference `check` is held to.

One prefill is batch 1, no history: `seq_len` prompt tokens in, the logits of the
last position out (the first generated token is their argmax), together with the
state a decode step would continue from -- the K/V rows of the 16 attention layers
and the convolution window and recurrent matrix of the 48 GatedDeltaNet layers.

Every rounding point follows the published Hugging Face code
(`transformers.models.qwen3_5.modeling_qwen3_5`, the torch fallback path that runs
when `fla` / `causal_conv1d` are absent): matmuls land in bf16, the RMSNorms
compute in f32 and land once, SwiGLU and the output gates round after each bf16
op, the gated delta rule runs in f32 in the same chunked (C=64) WY form as
`torch_chunk_gated_delta_rule`. Weights keep the checkpoint's own (out, in)
layout, so every projection is `matmul(x, w, b_layout="NK")`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_CONFIG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
config = _CONFIG["text_config"]

_DT = {"bfloat16": "bf16"}[config["dtype"]]
_EPS = config["rms_norm_eps"]
_H = config["hidden_size"]
_I = config["intermediate_size"]
_V = config["vocab_size"]
_LAYER_TYPES = tuple(config["layer_types"])

# Gated attention.
_HQ = config["num_attention_heads"]
_HKV = config["num_key_value_heads"]
_D = config["head_dim"]
_G = _HQ // _HKV
_ROT = int(_D * config["rope_parameters"]["partial_rotary_factor"])
_HALF = _ROT // 2
_SCALE = _D ** -0.5
_NEG_INF = float("-inf")
_N_ATTN = _LAYER_TYPES.count("full_attention")

# Gated delta net.
_HK = config["linear_num_key_heads"]
_HV = config["linear_num_value_heads"]
_DK = config["linear_key_head_dim"]
_DV = config["linear_value_head_dim"]
_VPK = _HV // _HK
_KEY = _HK * _DK
_VAL = _HV * _DV
_CONV = 2 * _KEY + _VAL
_KERNEL = config["linear_conv_kernel_dim"]
_WINDOW = _KERNEL - 1
_QSCALE = 1.0 / math.sqrt(_DK)
_L2_EPS = 1e-6
_N_GDN = _LAYER_TYPES.count("linear_attention")

# The chunk length of the WY form; the prompt is zero-padded up to a whole chunk
# inside the delta rule only, exactly as the published fallback pads it.
_C = 64

#: Prompt length. 1024 is the longest prompt this round serves.
T = DimVar("seq_len", 1, 1024)
_NC = (T + _C - 1) // _C
_TP = _NC * _C
_TW = T + _WINDOW




@module(entry="layer")
class Qwen3_8DeltaNetLayer:
    """`Qwen3_5DecoderLayer` with a `Qwen3_5GatedDeltaNet` token mixer.

    One Module holds every weight of the layer; its helpers are same-Module
    functions handed their weights explicitly.
    """

    @func
    def swiglu(
        x: Tensor[(1, T, _H), _DT],
        w_gate: Tensor[(_I, _H), _DT],
        w_up: Tensor[(_I, _H), _DT],
        w_down: Tensor[(_H, _I), _DT],
    ) -> Tensor[(1, T, _H), _DT]:
        # HF `Qwen3_5MLP`: down(silu(gate(x)) * up(x)); silu and the product each
        # land in bf16.
        gate = tf.matmul(x, w_gate, b_layout="NK")
        up = tf.matmul(x, w_up, b_layout="NK")
        return tf.matmul(tf.silu(gate) * up, w_down, b_layout="NK")

    @func
    def causal_conv(
        mixed: Tensor[(1, T, _CONV), _DT],
        conv_w: Tensor[(_CONV, 1, _KERNEL), _DT],
    ) -> Tensor[(1, T, _CONV), _DT]:
        # Depthwise causal conv1d (left zero padding of kernel-1), accumulated in
        # f32, landed once in bf16, then silu: `F.silu(self.conv1d(x)[..., :T])`.
        padded = tf.insert_slice(tf.zeros(Tensor[(1, _TW, _CONV), _DT]), mixed, (0, _WINDOW, 0))
        taps = tf.cast(tf.reshape(conv_w, new_shape=(_CONV, _KERNEL)), dtype="f32")
        acc = tf.cast(padded[:, 0:T, :], dtype="f32") * tf.reshape(taps[:, 0:1], new_shape=(1, 1, _CONV))
        acc = acc + tf.cast(padded[:, 1 : T + 1, :], dtype="f32") * tf.reshape(taps[:, 1:2], new_shape=(1, 1, _CONV))
        acc = acc + tf.cast(padded[:, 2 : T + 2, :], dtype="f32") * tf.reshape(taps[:, 2:3], new_shape=(1, 1, _CONV))
        acc = acc + tf.cast(padded[:, 3 : T + 3, :], dtype="f32") * tf.reshape(taps[:, 3:4], new_shape=(1, 1, _CONV))
        return tf.silu(tf.cast(acc, dtype=_DT))

    @func
    def l2_normalise(x: Tensor[(1, T, _HK, _DK), _DT]) -> Tensor[(1, T, _HK, _DK), _DT]:
        # FLA-aligned `l2norm`, in the input's own dtype: rsqrt(sum(x*x) + eps) * x.
        square_sum = tf.reduce(tf.square(x), axes=(-1,), keepdim=True, kind="sum")
        return x * tf.rsqrt(square_sum + tf.full_like(square_sum, value=_L2_EPS))

    @func
    def delta_rule(
        q: Tensor[(1, T, _HV, _DK), _DT],
        k: Tensor[(1, T, _HV, _DK), _DT],
        v: Tensor[(1, T, _HV, _DV), _DT],
        g: Tensor[(1, T, _HV), "f32"],
        beta: Tensor[(1, T, _HV), _DT],
    ):
        # `torch_chunk_gated_delta_rule` with no initial state: f32 throughout,
        # chunks of 64, intra-chunk WY inverse by forward substitution, then one
        # sequential pass over the chunks carrying the recurrent matrix.
        # Returns the read-out (bf16, [1, T, HV, DV]) and the final state (f32).
        qf = tf.transpose(tf.cast(q, dtype="f32"), perm=(0, 2, 1, 3))
        kf = tf.transpose(tf.cast(k, dtype="f32"), perm=(0, 2, 1, 3))
        vf = tf.transpose(tf.cast(v, dtype="f32"), perm=(0, 2, 1, 3))
        bf = tf.transpose(tf.cast(beta, dtype="f32"), perm=(0, 2, 1))
        gf = tf.transpose(g, perm=(0, 2, 1))
        qp = tf.insert_slice(tf.zeros(Tensor[(1, _HV, _TP, _DK), "f32"]), qf, (0, 0, 0, 0))
        kp = tf.insert_slice(tf.zeros(Tensor[(1, _HV, _TP, _DK), "f32"]), kf, (0, 0, 0, 0))
        vp = tf.insert_slice(tf.zeros(Tensor[(1, _HV, _TP, _DV), "f32"]), vf, (0, 0, 0, 0))
        bp = tf.insert_slice(tf.zeros(Tensor[(1, _HV, _TP), "f32"]), bf, (0, 0, 0))
        gp = tf.insert_slice(tf.zeros(Tensor[(1, _HV, _TP), "f32"]), gf, (0, 0, 0))

        qs = qp * _QSCALE
        v_beta = vp * tf.reshape(bp, new_shape=(1, _HV, _TP, 1))
        k_beta = kp * tf.reshape(bp, new_shape=(1, _HV, _TP, 1))
        qc = tf.reshape(qs, new_shape=(1, _HV, _NC, _C, _DK))
        kc = tf.reshape(kp, new_shape=(1, _HV, _NC, _C, _DK))
        vbc = tf.reshape(v_beta, new_shape=(1, _HV, _NC, _C, _DV))
        kbc = tf.reshape(k_beta, new_shape=(1, _HV, _NC, _C, _DK))
        gc = tf.reshape(gp, new_shape=(1, _HV, _NC, _C))

        idx = tf.arange(Tensor[(_C,), "i64"])
        rows = tf.reshape(idx, new_shape=(_C, 1))
        cols = tf.reshape(idx, new_shape=(1, _C))
        lower = tf.cast(rows >= cols, dtype="f32")
        strict = tf.cast(rows > cols, dtype="f32")
        eye = tf.cast(rows == cols, dtype="f32")

        # In-chunk cumulative decay: g.cumsum(-1), as a product with the
        # upper-triangular ones (column i sums rows j <= i).
        gcum = tf.matmul(gc, tf.transpose(lower, perm=(1, 0)))
        diff = tf.reshape(gcum, new_shape=(1, _HV, _NC, _C, 1)) - tf.reshape(gcum, new_shape=(1, _HV, _NC, 1, _C))
        # tril before exp keeps the upper triangle finite, tril after zeroes it.
        decay = tf.exp(diff * lower) * lower

        attn = tf.neg(tf.matmul(kbc, tf.transpose(kc, perm=(0, 1, 2, 4, 3))) * decay) * strict
        # Forward substitution: row i += row i @ (rows < i, already updated).
        for i in range(1, _C):
            row = attn[:, :, :, i : i + 1, :]
            attn = tf.insert_slice(attn, row + tf.matmul(row, attn), (0, 0, 0, i, 0))
        attn = attn + eye

        value = tf.matmul(attn, vbc)
        k_cumdecay = tf.matmul(attn, kbc * tf.reshape(tf.exp(gcum), new_shape=(1, _HV, _NC, _C, 1)))

        state = tf.zeros(Tensor[(1, _HV, _DK, _DV), "f32"])
        out = tf.zeros(Tensor[(1, _HV, _NC, _C, _DV), "f32"])
        for c in range(_NC):
            q_i = tf.reshape(qc[:, :, c : c + 1, :, :], new_shape=(1, _HV, _C, _DK))
            k_i = tf.reshape(kc[:, :, c : c + 1, :, :], new_shape=(1, _HV, _C, _DK))
            v_i = tf.reshape(value[:, :, c : c + 1, :, :], new_shape=(1, _HV, _C, _DV))
            kcd_i = tf.reshape(k_cumdecay[:, :, c : c + 1, :, :], new_shape=(1, _HV, _C, _DK))
            g_i = tf.reshape(gcum[:, :, c : c + 1, :], new_shape=(1, _HV, _C))
            decay_i = tf.reshape(decay[:, :, c : c + 1, :, :], new_shape=(1, _HV, _C, _C))
            g_last = g_i[:, :, _C - 1 : _C]

            scores = tf.matmul(q_i, tf.transpose(k_i, perm=(0, 1, 3, 2))) * decay_i
            v_new = v_i - tf.matmul(kcd_i, state)
            inter = tf.matmul(q_i * tf.reshape(tf.exp(g_i), new_shape=(1, _HV, _C, 1)), state)
            o_i = inter + tf.matmul(scores, v_new)
            out = tf.insert_slice(out, tf.reshape(o_i, new_shape=(1, _HV, 1, _C, _DV)), (0, 0, c, 0, 0))
            k_w = k_i * tf.reshape(tf.exp(g_last - g_i), new_shape=(1, _HV, _C, 1))
            state = state * tf.reshape(tf.exp(g_last), new_shape=(1, _HV, 1, 1)) + tf.matmul(
                tf.transpose(k_w, perm=(0, 1, 3, 2)), v_new
            )

        flat = tf.reshape(out, new_shape=(1, _HV, _TP, _DV))
        read = tf.transpose(flat[:, :, 0:T, :], perm=(0, 2, 1, 3))
        return tf.cast(read, dtype=_DT), state

    @func
    def gated_delta_net(
        x: Tensor[(1, T, _H), _DT],
        w_qkv: Tensor[(_CONV, _H), _DT],
        w_z: Tensor[(_VAL, _H), _DT],
        w_b: Tensor[(_HV, _H), _DT],
        w_a: Tensor[(_HV, _H), _DT],
        conv_w: Tensor[(_CONV, 1, _KERNEL), _DT],
        a_log: Tensor[(_HV,), _DT],
        dt_bias: Tensor[(_HV,), _DT],
        gamma_gdn: Tensor[(_DV,), _DT],
        w_out: Tensor[(_H, _VAL), _DT],
    ):
        # `Qwen3_5GatedDeltaNet.forward` with no cache in, the cache it would write
        # out: returns (out_proj output, conv window [1, CONV, K-1], final state).
        mixed = tf.matmul(x, w_qkv, b_layout="NK")
        z = tf.matmul(x, w_z, b_layout="NK")
        b = tf.matmul(x, w_b, b_layout="NK")
        a = tf.matmul(x, w_a, b_layout="NK")

        # The last kernel-1 rows; a slice start written as T - 3 is not an operand
        # the specializer can bind, an index range over arange(start=T - 3) is.
        tail = tf.arange(Tensor[(_WINDOW,), "i64"], start=T - _WINDOW)
        window = tf.transpose(tf.index_select(mixed, tail, dim=1), perm=(0, 2, 1))
        conv = causal_conv(mixed, conv_w)

        # Normalise the 16 key heads, then let each of the 48 value heads read
        # its group's key head (normalising after the repeat is the same value).
        q = l2_normalise(tf.reshape(conv[:, :, 0:_KEY], new_shape=(1, T, _HK, _DK)))
        k = l2_normalise(tf.reshape(conv[:, :, _KEY : 2 * _KEY], new_shape=(1, T, _HK, _DK)))
        v = tf.reshape(conv[:, :, 2 * _KEY : _CONV], new_shape=(1, T, _HV, _DV))
        q = tf.repeat_interleave(q, repeats=_VPK, axis=2)
        k = tf.repeat_interleave(k, repeats=_VPK, axis=2)

        beta = tf.sigmoid(b)
        g = tf.neg(tf.exp(tf.cast(a_log, dtype="f32"))) * tf.softplus(
            tf.cast(a, dtype="f32") + tf.cast(dt_bias, dtype="f32")
        )
        read, state = delta_rule(q, k, v, g, beta)

        # `Qwen3_5RMSNormGated`: normalise per value head in f32, land in bf16,
        # scale by the bf16 weight, gate by silu(z) in f32, land in bf16.
        rf = tf.cast(read, dtype="f32")
        ms = tf.reduce(tf.square(rf), axes=(-1,), keepdim=True, kind="mean")
        normed = tf.cast(rf * tf.rsqrt(ms + _EPS), dtype=_DT) * gamma_gdn
        zf = tf.cast(tf.reshape(z, new_shape=(1, T, _HV, _DV)), dtype="f32")
        gated = tf.cast(tf.cast(normed, dtype="f32") * tf.silu(zf), dtype=_DT)
        out = tf.matmul(tf.reshape(gated, new_shape=(1, T, _VAL)), w_out, b_layout="NK")
        return out, window, state

    @func
    def layer(
        hidden: Tensor[(1, T, _H), _DT],
        gamma_in: ConstTensor[(_H,), _DT],
        w_qkv: ConstTensor[(_CONV, _H), _DT],
        w_z: ConstTensor[(_VAL, _H), _DT],
        w_b: ConstTensor[(_HV, _H), _DT],
        w_a: ConstTensor[(_HV, _H), _DT],
        conv_w: ConstTensor[(_CONV, 1, _KERNEL), _DT],
        a_log: ConstTensor[(_HV,), _DT],
        dt_bias: ConstTensor[(_HV,), _DT],
        gamma_gdn: ConstTensor[(_DV,), _DT],
        w_out: ConstTensor[(_H, _VAL), _DT],
        gamma_post: ConstTensor[(_H,), _DT],
        w_gate: ConstTensor[(_I, _H), _DT],
        w_up: ConstTensor[(_I, _H), _DT],
        w_down: ConstTensor[(_H, _I), _DT],
    ):
        # `Qwen3_5DecoderLayer` (linear_attention): pre-norm, mixer, residual,
        # pre-norm, MLP, residual. Returns (hidden, conv window, recurrent state).
        normed = tf.rms_norm(hidden, tf.cast(gamma_in, dtype="f32") + 1.0, eps=_EPS)
        mixed, window, state = gated_delta_net(
            normed, w_qkv, w_z, w_b, w_a, conv_w, a_log, dt_bias, gamma_gdn, w_out
        )
        attended = hidden + mixed
        post = tf.rms_norm(attended, tf.cast(gamma_post, dtype="f32") + 1.0, eps=_EPS)
        return attended + swiglu(post, w_gate, w_up, w_down), window, state


@module(entry="layer")
class Qwen3_8AttentionLayer:
    """`Qwen3_5DecoderLayer` with a gated `Qwen3_5Attention` token mixer."""

    @func
    def swiglu(
        x: Tensor[(1, T, _H), _DT],
        w_gate: Tensor[(_I, _H), _DT],
        w_up: Tensor[(_I, _H), _DT],
        w_down: Tensor[(_H, _I), _DT],
    ) -> Tensor[(1, T, _H), _DT]:
        # HF `Qwen3_5MLP`, as in the delta-net layer.
        gate = tf.matmul(x, w_gate, b_layout="NK")
        up = tf.matmul(x, w_up, b_layout="NK")
        return tf.matmul(tf.silu(gate) * up, w_down, b_layout="NK")

    @func
    def partial_rope(
        x: Tensor[(1, T, _HQ, _D), _DT],
        cos: Tensor[(T, _ROT), _DT],
        sin: Tensor[(T, _ROT), _DT],
    ) -> Tensor[(1, T, _HQ, _D), _DT]:
        # HF `apply_rotary_pos_emb` on the leading `rotary_dim` entries, in bf16:
        # (x * cos) + (rotate_half(x) * sin), each op landing in bf16.
        rot = x[:, :, :, 0:_ROT]
        rotated = tf.concat([tf.neg(rot[:, :, :, _HALF:_ROT]), rot[:, :, :, 0:_HALF]], axis=-1)
        c = tf.reshape(cos, new_shape=(1, T, 1, _ROT))
        s = tf.reshape(sin, new_shape=(1, T, 1, _ROT))
        return tf.concat([rot * c + rotated * s, x[:, :, :, _ROT:_D]], axis=-1)

    @func
    def partial_rope_kv(
        x: Tensor[(1, T, _HKV, _D), _DT],
        cos: Tensor[(T, _ROT), _DT],
        sin: Tensor[(T, _ROT), _DT],
    ) -> Tensor[(1, T, _HKV, _D), _DT]:
        # The same rotation over the key's head count (a Function's shapes are fixed).
        rot = x[:, :, :, 0:_ROT]
        rotated = tf.concat([tf.neg(rot[:, :, :, _HALF:_ROT]), rot[:, :, :, 0:_HALF]], axis=-1)
        c = tf.reshape(cos, new_shape=(1, T, 1, _ROT))
        s = tf.reshape(sin, new_shape=(1, T, 1, _ROT))
        return tf.concat([rot * c + rotated * s, x[:, :, :, _ROT:_D]], axis=-1)

    @func
    def attend(
        q: Tensor[(1, T, _HQ, _D), _DT],
        k: Tensor[(1, T, _HKV, _D), _DT],
        v: Tensor[(1, T, _HKV, _D), _DT],
    ) -> Tensor[(1, T, _HQ, _D), _DT]:
        # Causal GQA softmax attention as the flash kernels compute it: scores in
        # f32, softmax statistics in f32, unnormalised probabilities landed in bf16
        # for the PV product, f32 accumulation, normalised and landed once.
        qh = tf.transpose(q, perm=(0, 2, 1, 3))
        kh = tf.transpose(tf.repeat_interleave(k, repeats=_G, axis=2), perm=(0, 2, 1, 3))
        vh = tf.transpose(tf.repeat_interleave(v, repeats=_G, axis=2), perm=(0, 2, 1, 3))
        scores = tf.matmul(qh, kh, b_layout="NK", out_dtype="f32") * _SCALE
        pos = tf.arange(Tensor[(T,), "i64"])
        allowed = tf.reshape(pos, new_shape=(1, 1, 1, T)) <= tf.reshape(pos, new_shape=(1, 1, T, 1))
        masked = tf.where(allowed, scores, tf.full_like(scores, value=_NEG_INF))
        peak = tf.reduce(masked, axes=(-1,), keepdim=True, kind="max")
        p = tf.exp(masked - peak)
        total = tf.reduce(p, axes=(-1,), keepdim=True, kind="sum")
        acc = tf.matmul(tf.cast(p, dtype=_DT), vh, out_dtype="f32")
        return tf.transpose(tf.cast(acc / total, dtype=_DT), perm=(0, 2, 1, 3))

    @func
    def gated_attention(
        x: Tensor[(1, T, _H), _DT],
        cos: Tensor[(T, _ROT), _DT],
        sin: Tensor[(T, _ROT), _DT],
        w_q: Tensor[(_HQ * _D * 2, _H), _DT],
        w_k: Tensor[(_HKV * _D, _H), _DT],
        w_v: Tensor[(_HKV * _D, _H), _DT],
        gamma_q: Tensor[(_D,), _DT],
        gamma_k: Tensor[(_D,), _DT],
        w_o: Tensor[(_H, _HQ * _D), _DT],
    ):
        # `Qwen3_5Attention.forward`, no cache in: returns (o_proj output, this
        # prompt's K rows after norm + rope, V rows), the K/V a decode appends to.
        # q_proj's fan-out is [head, 2 * head_dim]: query then output gate per head.
        qg = tf.reshape(tf.matmul(x, w_q, b_layout="NK"), new_shape=(1, T, _HQ, 2 * _D))
        q = tf.rms_norm(qg[:, :, :, 0:_D], tf.cast(gamma_q, dtype="f32") + 1.0, eps=_EPS)
        gate = tf.reshape(qg[:, :, :, _D : 2 * _D], new_shape=(1, T, _HQ * _D))
        k = tf.rms_norm(
            tf.reshape(tf.matmul(x, w_k, b_layout="NK"), new_shape=(1, T, _HKV, _D)),
            tf.cast(gamma_k, dtype="f32") + 1.0,
            eps=_EPS,
        )
        v = tf.reshape(tf.matmul(x, w_v, b_layout="NK"), new_shape=(1, T, _HKV, _D))
        q_rope = partial_rope(q, cos, sin)
        k_rope = partial_rope_kv(k, cos, sin)
        attn = tf.reshape(attend(q_rope, k_rope, v), new_shape=(1, T, _HQ * _D))
        out = tf.matmul(attn * tf.sigmoid(gate), w_o, b_layout="NK")
        return out, k_rope, v

    @func
    def layer(
        hidden: Tensor[(1, T, _H), _DT],
        cos: Tensor[(T, _ROT), _DT],
        sin: Tensor[(T, _ROT), _DT],
        gamma_in: ConstTensor[(_H,), _DT],
        w_q: ConstTensor[(_HQ * _D * 2, _H), _DT],
        w_k: ConstTensor[(_HKV * _D, _H), _DT],
        w_v: ConstTensor[(_HKV * _D, _H), _DT],
        gamma_q: ConstTensor[(_D,), _DT],
        gamma_k: ConstTensor[(_D,), _DT],
        w_o: ConstTensor[(_H, _HQ * _D), _DT],
        gamma_post: ConstTensor[(_H,), _DT],
        w_gate: ConstTensor[(_I, _H), _DT],
        w_up: ConstTensor[(_I, _H), _DT],
        w_down: ConstTensor[(_H, _I), _DT],
    ):
        # `Qwen3_5DecoderLayer` (full_attention). Returns (hidden, K rows, V rows).
        normed = tf.rms_norm(hidden, tf.cast(gamma_in, dtype="f32") + 1.0, eps=_EPS)
        mixed, k, v = gated_attention(normed, cos, sin, w_q, w_k, w_v, gamma_q, gamma_k, w_o)
        attended = hidden + mixed
        post = tf.rms_norm(attended, tf.cast(gamma_post, dtype="f32") + 1.0, eps=_EPS)
        return attended + swiglu(post, w_gate, w_up, w_down), k, v


_LAYER = {"linear_attention": Qwen3_8DeltaNetLayer, "full_attention": Qwen3_8AttentionLayer}


@module(
    entry="prefill",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 384)),
)
class Qwen3_8_27B:
    """The text model in `layer_types` order and the step around it."""

    layers = tuple(_LAYER[kind].renamed(f"layer{i}") for i, kind in enumerate(_LAYER_TYPES))

    @func
    def embed(
        token_ids: Tensor[(T,), "i64"],
        table: ConstTensor[(_V, _H), _DT],
    ) -> Tensor[(1, T, _H), _DT]:
        # `embed_tokens`: one row per prompt token.
        return tf.reshape(tf.index_select(table, token_ids, dim=0), new_shape=(1, T, _H))

    @func
    def final_norm(
        hidden: Tensor[(1, T, _H), _DT],
        gamma_final: ConstTensor[(_H,), _DT],
    ) -> Tensor[(1, T, _H), _DT]:
        # `Qwen3_5TextModel.norm` (a Qwen3_5RMSNorm: scale by 1 + weight).
        return tf.rms_norm(hidden, tf.cast(gamma_final, dtype="f32") + 1.0, eps=_EPS)

    @func
    def lm_head(
        hidden: Tensor[(1, T, _H), _DT],
        w_head: ConstTensor[(_V, _H), _DT],
    ) -> Tensor[(1, _V), _DT]:
        # `lm_head` over the last position only: its argmax is the first token.
        last = tf.reshape(tf.index_select(hidden, tf.arange(Tensor[(1,), "i64"], start=T - 1), dim=1), new_shape=(1, _H))
        return tf.matmul(last, w_head, b_layout="NK")

    @func
    def prefill(
        token_ids: Tensor[(T,), "i64"],
        cos: Tensor[(T, _ROT), _DT],
        sin: Tensor[(T, _ROT), _DT],
        table: ConstTensor[(_V, _H), _DT],
        gamma_final: ConstTensor[(_H,), _DT],
        w_head: ConstTensor[(_V, _H), _DT],
    ):
        # The whole prefill: embedding, the 64 layers, the final norm, the head.
        # Returns (logits [1, V], K cache [16, T, 4, 256], V cache [16, T, 4, 256],
        # conv windows [48, 10240, 3], recurrent states [48, 48, 128, 128] f32).
        h = embed(token_ids, table)
        h, conv0, rec0 = layer0.layer(h)
        h, conv1, rec1 = layer1.layer(h)
        h, conv2, rec2 = layer2.layer(h)
        h, k3, v3 = layer3.layer(h, cos, sin)
        h, conv4, rec4 = layer4.layer(h)
        h, conv5, rec5 = layer5.layer(h)
        h, conv6, rec6 = layer6.layer(h)
        h, k7, v7 = layer7.layer(h, cos, sin)
        h, conv8, rec8 = layer8.layer(h)
        h, conv9, rec9 = layer9.layer(h)
        h, conv10, rec10 = layer10.layer(h)
        h, k11, v11 = layer11.layer(h, cos, sin)
        h, conv12, rec12 = layer12.layer(h)
        h, conv13, rec13 = layer13.layer(h)
        h, conv14, rec14 = layer14.layer(h)
        h, k15, v15 = layer15.layer(h, cos, sin)
        h, conv16, rec16 = layer16.layer(h)
        h, conv17, rec17 = layer17.layer(h)
        h, conv18, rec18 = layer18.layer(h)
        h, k19, v19 = layer19.layer(h, cos, sin)
        h, conv20, rec20 = layer20.layer(h)
        h, conv21, rec21 = layer21.layer(h)
        h, conv22, rec22 = layer22.layer(h)
        h, k23, v23 = layer23.layer(h, cos, sin)
        h, conv24, rec24 = layer24.layer(h)
        h, conv25, rec25 = layer25.layer(h)
        h, conv26, rec26 = layer26.layer(h)
        h, k27, v27 = layer27.layer(h, cos, sin)
        h, conv28, rec28 = layer28.layer(h)
        h, conv29, rec29 = layer29.layer(h)
        h, conv30, rec30 = layer30.layer(h)
        h, k31, v31 = layer31.layer(h, cos, sin)
        h, conv32, rec32 = layer32.layer(h)
        h, conv33, rec33 = layer33.layer(h)
        h, conv34, rec34 = layer34.layer(h)
        h, k35, v35 = layer35.layer(h, cos, sin)
        h, conv36, rec36 = layer36.layer(h)
        h, conv37, rec37 = layer37.layer(h)
        h, conv38, rec38 = layer38.layer(h)
        h, k39, v39 = layer39.layer(h, cos, sin)
        h, conv40, rec40 = layer40.layer(h)
        h, conv41, rec41 = layer41.layer(h)
        h, conv42, rec42 = layer42.layer(h)
        h, k43, v43 = layer43.layer(h, cos, sin)
        h, conv44, rec44 = layer44.layer(h)
        h, conv45, rec45 = layer45.layer(h)
        h, conv46, rec46 = layer46.layer(h)
        h, k47, v47 = layer47.layer(h, cos, sin)
        h, conv48, rec48 = layer48.layer(h)
        h, conv49, rec49 = layer49.layer(h)
        h, conv50, rec50 = layer50.layer(h)
        h, k51, v51 = layer51.layer(h, cos, sin)
        h, conv52, rec52 = layer52.layer(h)
        h, conv53, rec53 = layer53.layer(h)
        h, conv54, rec54 = layer54.layer(h)
        h, k55, v55 = layer55.layer(h, cos, sin)
        h, conv56, rec56 = layer56.layer(h)
        h, conv57, rec57 = layer57.layer(h)
        h, conv58, rec58 = layer58.layer(h)
        h, k59, v59 = layer59.layer(h, cos, sin)
        h, conv60, rec60 = layer60.layer(h)
        h, conv61, rec61 = layer61.layer(h)
        h, conv62, rec62 = layer62.layer(h)
        h, k63, v63 = layer63.layer(h, cos, sin)
        logits = lm_head(final_norm(h, gamma_final), w_head)
        k_cache = tf.reshape(
            tf.stack([k3, k7, k11, k15, k19, k23, k27, k31, k35, k39, k43, k47, k51, k55, k59, k63], axis=0),
            new_shape=(_N_ATTN, T, _HKV, _D),
        )
        v_cache = tf.reshape(
            tf.stack([v3, v7, v11, v15, v19, v23, v27, v31, v35, v39, v43, v47, v51, v55, v59, v63], axis=0),
            new_shape=(_N_ATTN, T, _HKV, _D),
        )
        conv_state = tf.reshape(
            tf.stack([conv0, conv1, conv2, conv4, conv5, conv6, conv8, conv9, conv10, conv12, conv13, conv14, conv16, conv17, conv18, conv20, conv21, conv22, conv24, conv25, conv26, conv28, conv29, conv30, conv32, conv33, conv34, conv36, conv37, conv38, conv40, conv41, conv42, conv44, conv45, conv46, conv48, conv49, conv50, conv52, conv53, conv54, conv56, conv57, conv58, conv60, conv61, conv62], axis=0),
            new_shape=(_N_GDN, _CONV, _WINDOW),
        )
        rec_state = tf.reshape(
            tf.stack([rec0, rec1, rec2, rec4, rec5, rec6, rec8, rec9, rec10, rec12, rec13, rec14, rec16, rec17, rec18, rec20, rec21, rec22, rec24, rec25, rec26, rec28, rec29, rec30, rec32, rec33, rec34, rec36, rec37, rec38, rec40, rec41, rec42, rec44, rec45, rec46, rec48, rec49, rec50, rec52, rec53, rec54, rec56, rec57, rec58, rec60, rec61, rec62], axis=0),
            new_shape=(_N_GDN, _HV, _DK, _DV),
        )
        return logits, k_cache, v_cache, conv_state, rec_state
