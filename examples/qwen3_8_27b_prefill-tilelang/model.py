"""Qwen3.8-27B text prefill as authored HIR: batch 1, no history, first token only.

This is the reference `check` holds every implementation to. It mirrors the
published `transformers` classes (`Qwen3_5TextModel`, `Qwen3_5DecoderLayer`,
`Qwen3_5GatedDeltaNet`, `Qwen3_5Attention`, `Qwen3_5MLP`) one function per
published step, at the precision each step lands in:

- `Qwen3_5RMSNorm` normalises in f32, scales by `1 + weight` in f32, then
  lands in bf16 once; `tf.rms_norm` with an f32 weight is exactly that.
- `Qwen3_5RMSNormGated` lands in bf16 *before* its bf16 weight and multiplies
  the f32 `silu(gate)` afterwards, so it is written out step by step.
- The gated delta rule is the published chunked form (`chunk_size = 64`):
  per chunk a WY transform, then the recurrent state update across chunks.
- Attention scores stay in f32 through the softmax, as the fused kernels the
  published model dispatches to keep them.

Function boundaries are the fusion groups evidence/STEP3.md decided (v1; the
per-published-module v0 is model_v0.py): each function is what one runtime
kernel, or one tight group of them, implements, so the residual adds sit in
the GEMMs that produce their second operand, SwiGLU in the gate/up GEMM, the
output gate in attention, and conv + SiLU + l2norm + the gates in `prep`.

Every prompt length is a multiple of the 64-token chunk. A prompt of another
length is right-padded by the caller; everything here is causal, so the rows
before the padding are exactly the rows of the unpadded prompt, and `final`
reads the row the caller names.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_CONFIG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
config = _CONFIG["text_config"]

_DT = "bf16"
_EPS = config["rms_norm_eps"]

H = config["hidden_size"]
I = config["intermediate_size"]
V = config["vocab_size"]

# Full attention.
HQ = config["num_attention_heads"]
HKV = config["num_key_value_heads"]
D = config["head_dim"]
GQA = HQ // HKV
ROT = int(D * config["rope_parameters"]["partial_rotary_factor"])
HALF = ROT // 2
ROPE_THETA = config["rope_parameters"]["rope_theta"]
SCALE = D ** -0.5
QG = HQ * D * 2  # q_proj's fan-out: query and output gate, interleaved per head

# Gated delta net.
HK = config["linear_num_key_heads"]
HV = config["linear_num_value_heads"]
DK = config["linear_key_head_dim"]
DV = config["linear_value_head_dim"]
KEY = HK * DK
VAL = HV * DV
CONV = 2 * KEY + VAL
KERNEL = config["linear_conv_kernel_dim"]
VPK = HV // HK
QSCALE = DK ** -0.5
L2_EPS = 1e-6  # the FLA-aligned `l2norm` default, not a config field
CHUNK = 64  # `torch_chunk_gated_delta_rule(chunk_size=64)`

NEG_INF = float("-inf")

#: Prompt length. A multiple of CHUNK; the caller pads to one.
N = DimVar("seq_len", CHUNK, 8192)


@module(entry="down")
class MLP:
    """`Qwen3_5MLP` and the residual add that follows it."""

    @func
    def gate_up(
        hn: Tensor[(1, N, H), _DT],
        w_gate: ConstTensor[(I, H), _DT],
        w_up: ConstTensor[(I, H), _DT],
    ) -> Tensor[(1, N, I), _DT]:
        # SwiGLU: silu(gate) and its product with up each land in bf16.
        gate = tf.matmul(hn, w_gate, b_layout="NK")
        up = tf.matmul(hn, w_up, b_layout="NK")
        return tf.silu(gate) * up

    @func
    def down(
        act: Tensor[(1, N, I), _DT],
        hidden: Tensor[(1, N, H), _DT],
        w_down: ConstTensor[(H, I), _DT],
    ) -> Tensor[(1, N, H), _DT]:
        # down_proj lands in bf16, then the residual add lands again.
        return hidden + tf.matmul(act, w_down, b_layout="NK")


@module(entry="delta_rule")
class GatedDeltaNet:
    """`Qwen3_5GatedDeltaNet` over a whole prompt, from an empty state."""

    @func
    def in_proj(
        hn: Tensor[(1, N, H), _DT],
        w_qkv: ConstTensor[(CONV, H), _DT],
        w_z: ConstTensor[(VAL, H), _DT],
        w_b: ConstTensor[(HV, H), _DT],
        w_a: ConstTensor[(HV, H), _DT],
    ):
        # The four input projections. `qkv` feeds the convolution, `z` the
        # output gate, `b` and `a` the per-head write strength and decay.
        qkv = tf.matmul(hn, w_qkv, b_layout="NK")
        z = tf.matmul(hn, w_z, b_layout="NK")
        b = tf.matmul(hn, w_b, b_layout="NK")
        a = tf.matmul(hn, w_a, b_layout="NK")
        return qkv, z, b, a

    @func
    def prep(
        qkv: Tensor[(1, N, CONV), _DT],
        b: Tensor[(1, N, HV), _DT],
        a: Tensor[(1, N, HV), _DT],
        conv_w: ConstTensor[(CONV, 1, KERNEL), _DT],
        a_log: ConstTensor[(HV,), _DT],
        dt_bias: ConstTensor[(HV,), _DT],
    ):
        # Depthwise causal convolution, zero history: output t sums taps
        # j = 0..3 against input t - 3 + j. Accumulate in f32, land in bf16
        # once, then SiLU -- `F.silu(self.conv1d(x)[..., :N])` in bf16.
        taps = tf.cast(
            tf.transpose(tf.reshape(conv_w, new_shape=(CONV, KERNEL)), perm=(1, 0)),
            dtype="f32",
        )
        x = tf.cast(qkv, dtype="f32")
        acc = x * taps[3:4, :]
        acc = tf.insert_slice(acc, acc[:, 1:N, :] + x[:, 0:N - 1, :] * taps[2:3, :], (0, 1, 0))
        acc = tf.insert_slice(acc, acc[:, 2:N, :] + x[:, 0:N - 2, :] * taps[1:2, :], (0, 2, 0))
        acc = tf.insert_slice(acc, acc[:, 3:N, :] + x[:, 0:N - 3, :] * taps[0:1, :], (0, 3, 0))
        mixed = tf.silu(tf.cast(acc, dtype=_DT))
        # Split into the per-key-head query and key and the per-value-head
        # value. q and k are l2-normalised as FLA's `l2norm`, run at the input
        # dtype: rsqrt of the sum of squares plus eps, every step landing in
        # bf16. Per head, so the same before or after the value heads repeat
        # the key heads.
        qk = tf.reshape(mixed[:, :, 0:2 * KEY], new_shape=(1, N, 2 * HK, DK))
        square_sum = tf.reduce(qk * qk, axes=(-1,), keepdim=True, kind="sum")
        qk = qk * tf.rsqrt(square_sum + tf.full_like(square_sum, value=L2_EPS))
        q = qk[:, :, 0:HK, :]
        k = qk[:, :, HK:2 * HK, :]
        v = tf.reshape(mixed[:, :, 2 * KEY:CONV], new_shape=(1, N, HV, DV))
        # beta = sigmoid(b) in bf16; g = -exp(A_log) * softplus(a + dt_bias)
        # in f32. g < 0, so exp(g) is a decay in (0, 1).
        beta = tf.sigmoid(b)
        g = -tf.exp(tf.cast(a_log, dtype="f32")) * tf.softplus(
            tf.cast(a, dtype="f32") + tf.cast(dt_bias, dtype="f32")
        )
        return q, k, v, g, beta

    @func
    def delta_rule(
        q: Tensor[(1, N, HK, DK), _DT],
        k: Tensor[(1, N, HK, DK), _DT],
        v: Tensor[(1, N, HV, DV), _DT],
        g: Tensor[(1, N, HV), "f32"],
        beta: Tensor[(1, N, HV), _DT],
    ) -> Tensor[(1, N, HV, DV), _DT]:
        # `torch_chunk_gated_delta_rule` on already-normalised q and k: every
        # value head reads the key head it shares.
        q = tf.repeat_interleave(q, repeats=VPK, axis=2)
        k = tf.repeat_interleave(k, repeats=VPK, axis=2)

        # Head-major f32, as the published routine transposes and widens.
        qh = tf.transpose(tf.cast(q, dtype="f32"), perm=(0, 2, 1, 3)) * QSCALE
        kh = tf.transpose(tf.cast(k, dtype="f32"), perm=(0, 2, 1, 3))
        vh = tf.transpose(tf.cast(v, dtype="f32"), perm=(0, 2, 1, 3))
        bh = tf.transpose(tf.cast(beta, dtype="f32"), perm=(0, 2, 1))
        gh = tf.transpose(g, perm=(0, 2, 1))

        # Chunk-local constants: inclusive prefix sum as a matmul against an
        # upper-triangular ones matrix, the causal masks, the identity.
        idx = tf.arange(Tensor[(CHUNK,), "i32"])
        row = tf.reshape(idx, new_shape=(CHUNK, 1))
        col = tf.reshape(idx, new_shape=(1, CHUNK))
        prefix = tf.cast(row <= col, dtype="f32")
        lower = row >= col
        strict = row > col
        eye = tf.cast(row == col, dtype="f32")
        zero_tile = tf.zeros(Tensor[(1, HV, CHUNK, CHUNK), "f32"])

        state = tf.zeros(Tensor[(1, HV, DK, DV), "f32"])
        out = tf.zeros(Tensor[(1, HV, N, DV), "f32"])
        for c in tf.tile(N, CHUNK):
            qc = qh[:, :, c, :]
            kc = kh[:, :, c, :]
            vc = vh[:, :, c, :]
            bc = tf.reshape(bh[:, :, c], new_shape=(1, HV, CHUNK, 1))
            gcum = tf.matmul(tf.reshape(gh[:, :, c], new_shape=(1, HV, 1, CHUNK)), prefix)
            g_col = tf.reshape(gcum, new_shape=(1, HV, CHUNK, 1))
            decay = tf.where(lower, tf.exp(g_col - gcum), zero_tile)

            k_beta = kc * bc
            v_beta = vc * bc
            # WY transform: T = (I + strict_lower(diag(beta) K K^T * decay))^-1
            # by the published forward substitution, one row at a time.
            neg = tf.where(strict, -(tf.matmul(k_beta, kc, b_layout="NK") * decay), zero_tile)
            for i in range(1, CHUNK):
                r = neg[:, :, i:i + 1, :]
                neg = tf.insert_slice(neg, r + tf.matmul(r, neg), (0, 0, i, 0))
            t = neg + eye
            u = tf.matmul(t, v_beta)
            w = tf.matmul(t, k_beta * tf.exp(g_col))

            # Across chunks: correct the values against the carried state,
            # read out, then decay and update the state.
            scores = tf.matmul(qc, kc, b_layout="NK") * decay
            v_new = u - tf.matmul(w, state)
            o = tf.matmul(qc * tf.exp(g_col), state) + tf.matmul(scores, v_new)
            g_last = gcum[:, :, :, CHUNK - 1:CHUNK]
            k_decay = kc * tf.exp(tf.reshape(g_last, new_shape=(1, HV, 1, 1)) - g_col)
            state = state * tf.exp(tf.reshape(g_last, new_shape=(1, HV, 1, 1))) + tf.matmul(
                k_decay, v_new, a_layout="KM"
            )
            out = tf.insert_slice(out, o, (0, 0, c, 0))
        return tf.cast(tf.transpose(out, perm=(0, 2, 1, 3)), dtype=_DT)

    @func
    def gated_norm(
        core: Tensor[(1, N, HV, DV), _DT],
        z: Tensor[(1, N, VAL), _DT],
        gamma_gdn: ConstTensor[(DV,), _DT],
    ) -> Tensor[(1, N, VAL), _DT]:
        # `Qwen3_5RMSNormGated`: normalise per head in f32, land in bf16,
        # scale by the bf16 weight, multiply by silu(z) in f32, land again.
        unit = tf.full_like(tf.cast(gamma_gdn, dtype="f32"), value=1.0)
        normed = tf.rms_norm(core, unit, eps=_EPS) * gamma_gdn
        gate = tf.silu(tf.cast(tf.reshape(z, new_shape=(1, N, HV, DV)), dtype="f32"))
        gated = tf.cast(tf.cast(normed, dtype="f32") * gate, dtype=_DT)
        return tf.reshape(gated, new_shape=(1, N, VAL))

    @func
    def out_proj(
        gated: Tensor[(1, N, VAL), _DT],
        hidden: Tensor[(1, N, H), _DT],
        w_out: ConstTensor[(H, VAL), _DT],
    ) -> Tensor[(1, N, H), _DT]:
        # out_proj lands in bf16, then the residual add lands again.
        return hidden + tf.matmul(gated, w_out, b_layout="NK")


@module(entry="o_proj")
class Attention:
    """`Qwen3_5Attention` over a whole prompt: causal, gated output."""

    @func
    def qkv(
        hn: Tensor[(1, N, H), _DT],
        cos: Tensor[(N, ROT), _DT],
        sin: Tensor[(N, ROT), _DT],
        w_q: ConstTensor[(QG, H), _DT],
        w_k: ConstTensor[(HKV * D, H), _DT],
        w_v: ConstTensor[(HKV * D, H), _DT],
        gamma_q: ConstTensor[(D,), _DT],
        gamma_k: ConstTensor[(D,), _DT],
    ):
        # One q projection, two halves per head: the query and the output
        # gate (kept per head). q and k are RMS-normed per head, then rotated.
        qg = tf.reshape(tf.matmul(hn, w_q, b_layout="NK"), new_shape=(1, N, HQ, 2 * D))
        q = tf.rms_norm(qg[:, :, :, 0:D], tf.cast(gamma_q, dtype="f32") + 1.0, eps=_EPS)
        gate = qg[:, :, :, D:2 * D]
        k = tf.rms_norm(
            tf.reshape(tf.matmul(hn, w_k, b_layout="NK"), new_shape=(1, N, HKV, D)),
            tf.cast(gamma_k, dtype="f32") + 1.0,
            eps=_EPS,
        )
        v = tf.reshape(tf.matmul(hn, w_v, b_layout="NK"), new_shape=(1, N, HKV, D))
        # `apply_rotary_pos_emb` on the leading ROT entries of each head,
        # rotate-half form, each product and the sum landing in bf16.
        c = tf.reshape(cos, new_shape=(1, N, 1, ROT))
        s = tf.reshape(sin, new_shape=(1, N, 1, ROT))
        q_rot = q[:, :, :, 0:ROT]
        q_half = tf.concat([-q_rot[:, :, :, HALF:ROT], q_rot[:, :, :, 0:HALF]], axis=-1)
        q = tf.concat([q_rot * c + q_half * s, q[:, :, :, ROT:D]], axis=-1)
        k_rot = k[:, :, :, 0:ROT]
        k_half = tf.concat([-k_rot[:, :, :, HALF:ROT], k_rot[:, :, :, 0:HALF]], axis=-1)
        k = tf.concat([k_rot * c + k_half * s, k[:, :, :, ROT:D]], axis=-1)
        return q, gate, k, v

    @func
    def attention(
        q: Tensor[(1, N, HQ, D), _DT],
        k: Tensor[(1, N, HKV, D), _DT],
        v: Tensor[(1, N, HKV, D), _DT],
        gate: Tensor[(1, N, HQ, D), _DT],
    ) -> Tensor[(1, N, HQ * D), _DT]:
        # Causal softmax attention; every query head reads its group's key
        # and value head. Scores and softmax in f32, weights land in bf16
        # for the value product. Then the output gate, in bf16.
        qh = tf.transpose(q, perm=(0, 2, 1, 3))
        kh = tf.transpose(tf.repeat_interleave(k, repeats=GQA, axis=2), perm=(0, 2, 1, 3))
        vh = tf.transpose(tf.repeat_interleave(v, repeats=GQA, axis=2), perm=(0, 2, 1, 3))
        scores = tf.matmul(qh, kh, b_layout="NK", out_dtype="f32") * SCALE
        pos = tf.arange(Tensor[(N,), "i32"])
        causal = tf.reshape(pos, new_shape=(N, 1)) >= tf.reshape(pos, new_shape=(1, N))
        probs = tf.softmax(tf.where(causal, scores, tf.full_like(scores, value=NEG_INF)), axis=-1)
        o = tf.matmul(tf.cast(probs, dtype=_DT), vh)
        attn = tf.reshape(tf.transpose(o, perm=(0, 2, 1, 3)), new_shape=(1, N, HQ * D))
        return attn * tf.sigmoid(tf.reshape(gate, new_shape=(1, N, HQ * D)))

    @func
    def o_proj(
        gated: Tensor[(1, N, HQ * D), _DT],
        hidden: Tensor[(1, N, H), _DT],
        w_o: ConstTensor[(H, HQ * D), _DT],
    ) -> Tensor[(1, N, H), _DT]:
        # o_proj lands in bf16, then the residual add lands again.
        return hidden + tf.matmul(gated, w_o, b_layout="NK")


def _norms():
    """The two pre-norms both decoder layer types share."""

    @func
    def input_norm(
        hidden: Tensor[(1, N, H), _DT],
        gamma_in: ConstTensor[(H,), _DT],
    ) -> Tensor[(1, N, H), _DT]:
        return tf.rms_norm(hidden, tf.cast(gamma_in, dtype="f32") + 1.0, eps=_EPS)

    @func
    def post_norm(
        hidden: Tensor[(1, N, H), _DT],
        gamma_post: ConstTensor[(H,), _DT],
    ) -> Tensor[(1, N, H), _DT]:
        return tf.rms_norm(hidden, tf.cast(gamma_post, dtype="f32") + 1.0, eps=_EPS)

    return input_norm, post_norm


def _linear_layer_forward(self, hidden):
    """`Qwen3_5DecoderLayer.forward` for a linear-attention layer."""
    mixer = self.mixer
    qkv, z, b, a = mixer.in_proj(self.input_norm(hidden))
    q, k, v, g, beta = mixer.prep(qkv, b, a)
    core = mixer.delta_rule(q, k, v, g, beta)
    hidden = mixer.out_proj(mixer.gated_norm(core, z), hidden)
    return self.mlp.down(self.mlp.gate_up(self.post_norm(hidden)), hidden)


def _full_layer_forward(self, hidden, cos, sin):
    """`Qwen3_5DecoderLayer.forward` for a full-attention layer."""
    mixer = self.mixer
    q, gate, k, v = mixer.qkv(self.input_norm(hidden), cos, sin)
    hidden = mixer.o_proj(mixer.attention(q, k, v, gate), hidden)
    return self.mlp.down(self.mlp.gate_up(self.post_norm(hidden)), hidden)


@module(entry="input_norm")
class LinearAttentionLayer:
    mixer = GatedDeltaNet.renamed("mixer")
    mlp = MLP.renamed("mlp")
    input_norm, post_norm = _norms()
    forward = _linear_layer_forward


@module(entry="input_norm")
class FullAttentionLayer:
    mixer = Attention.renamed("mixer")
    mlp = MLP.renamed("mlp")
    input_norm, post_norm = _norms()
    forward = _full_layer_forward


LAYER_TYPE = {
    "linear_attention": LinearAttentionLayer,
    "full_attention": FullAttentionLayer,
}


def rope_tables(n: int, device):
    """`Qwen3_5TextRotaryEmbedding` for text positions 0..n-1, as bf16 rows.

    Text tokens carry one position on all three mrope axes, so the
    interleaved sections collapse to the plain rotary frequencies.
    """
    import torch  # noqa: PLC0415

    inv = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROT, 2, dtype=torch.int64, device=device).float() / ROT)
    )
    freqs = torch.arange(n, device=device).float()[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


@module(
    entry="final",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 132), Topology("thread", 384)),
)
class Qwen3_8_27B:
    """The text decoder in `config.layer_types` order, one prefill step."""

    layers = tuple(
        LAYER_TYPE[kind].renamed(f"layer{index}")
        for index, kind in enumerate(config["layer_types"])
    )

    @func
    def embed(
        table: ConstTensor[(V, H), _DT],
        token_ids: Tensor[(N,), "i64"],
    ) -> Tensor[(1, N, H), _DT]:
        return tf.reshape(tf.index_select(table, token_ids, dim=0), new_shape=(1, N, H))

    @func
    def final(
        hidden: Tensor[(1, N, H), _DT],
        last: Tensor[(1,), "i64"],
        gamma_final: ConstTensor[(H,), _DT],
        w_head: ConstTensor[(V, H), _DT],
    ) -> Tensor[(1, V), _DT]:
        # The closing norm and the head, for the one row the first token is
        # sampled from.
        row = tf.reshape(tf.index_select(hidden, last, dim=1), new_shape=(1, H))
        normed = tf.rms_norm(row, tf.cast(gamma_final, dtype="f32") + 1.0, eps=_EPS)
        return tf.matmul(normed, w_head, b_layout="NK")

    def forward(self, token_ids, last):
        """One prefill: token ids in, the logits of row `last` out."""
        hidden = self.embed(token_ids)
        cos, sin = rope_tables(token_ids.shape[0], token_ids.device)
        for layer, kind in zip(self.modules, config["layer_types"]):
            if kind == "full_attention":
                hidden = layer.forward(hidden, cos, sin)
            else:
                hidden = layer.forward(hidden)
        return self.final(hidden, last)


def hf_alias() -> dict[str, object]:
    """Canonical weight names -> the published checkpoint's own names.

    Every weight is read as stored: projections stay (out, in) and the
    matmuls say so with `b_layout="NK"`, the convolution keeps its
    (channels, 1, kernel) shape.
    """
    alias: dict[str, object] = {
        "table": "model.language_model.embed_tokens.weight",
        "gamma_final": "model.language_model.norm.weight",
        "w_head": "lm_head.weight",
        "gamma_in": "input_layernorm.weight",
        "gamma_post": "post_attention_layernorm.weight",
        "w_qkv": "in_proj_qkv.weight",
        "w_z": "in_proj_z.weight",
        "w_b": "in_proj_b.weight",
        "w_a": "in_proj_a.weight",
        "conv_w": "conv1d.weight",
        "a_log": "A_log",
        "dt_bias": "dt_bias",
        "gamma_gdn": "norm.weight",
        "w_out": "out_proj.weight",
        "w_q": "q_proj.weight",
        "w_k": "k_proj.weight",
        "w_v": "v_proj.weight",
        "w_o": "o_proj.weight",
        "gamma_q": "q_norm.weight",
        "gamma_k": "k_norm.weight",
        "w_gate": "gate_proj.weight",
        "w_up": "up_proj.weight",
        "w_down": "down_proj.weight",
    }
    for index, kind in enumerate(config["layer_types"]):
        alias[f"layer{index}"] = f"model.language_model.layers.{index}"
        alias[f"model.language_model.layers.{index}.mixer"] = (
            "linear_attn" if kind == "linear_attention" else "self_attn"
        )
    return alias
