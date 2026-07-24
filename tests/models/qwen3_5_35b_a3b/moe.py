"""Qwen3.5-35B-A3B MoE component (every decoder layer): 256 routed experts
top-8 + one scalar-gated shared expert. Mirrors
``transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`` (transformers
5.12.1, cited below as M), M:776/795.

``moe_mix``: post-attention RMSNorm -> router logits -> softmax (f32, all
256 experts) -> top-8 -> renormalize weights -> cast to bf16 -> each
selected expert computes ``silu(gate) * up -> down`` (experts stacked
``gate_up[256,1024,2048]`` / ``down[256,2048,512]``) -> weighted sum ->
shared expert (inter=512) scaled by a **scalar** sigmoid gate (``Linear
2048->1``, M:807/813) -> residual.

``moe_convert``: the sole RAW-checkpoint -> CANONICAL (kernel packed
layout) weight conversion entry point for ``moe_mix`` -- pure repack, no
numeric change. The shared expert is packed into slot ``N_EXPERTS`` of
``packed_gate_up`` / ``packed_down`` alongside the 256 routed experts.
"""
from __future__ import annotations

from tests.models.qwen3_5_35b_a3b.config import HIDDEN
from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.module import Module

N_EXPERTS = 256
TOP_K = 8
MOE_INTER = 512
PACKED_EXPERTS = N_EXPERTS + 1  # 256 routed + shared packed at slot N_EXPERTS


@func
def moe_mix(
    x: Tensor[(1, 1, HIDDEN), "bf16"],                       # layer residual stream (after the token mixer)
    post_norm_gamma_raw: ConstTensor[(HIDDEN,), "f32"],      # RAW (not +1); (1+w) applied in-body
    router_w: ConstTensor[(N_EXPERTS, HIDDEN), "bf16"],
    packed_gate_up: ConstTensor[(PACKED_EXPERTS, 2 * MOE_INTER, HIDDEN), "bf16"],
    packed_down: ConstTensor[(PACKED_EXPERTS, HIDDEN, MOE_INTER), "bf16"],
    shared_gate_w: ConstTensor[(1, HIDDEN), "bf16"],
) -> Tensor[(1, 1, HIDDEN), "bf16"]:
    # CANONICAL weight layout = the kernel's packed layout (sole source: see
    # moe_convert): the shared expert is packed into slot N_EXPERTS of
    # packed_gate_up/packed_down. Evaluator (this func) and the kernel
    # override read the same self.weights -- no second func-canonical weight
    # set to keep in sync.
    # ── post_attention_layernorm (M:871): RAW weight, (1+w) applied in-body ─
    h = tf.rms_norm(x, 1.0 + post_norm_gamma_raw)
    ht = tf.reshape(h, new_shape=(1, HIDDEN))

    # ── router (M:776): softmax(f32, all experts) -> top8 -> renormalize ────
    logits = tf.matmul(
        tf.cast(ht, dtype="f32"),
        tf.transpose(tf.cast(router_w, dtype="f32"), perm=(1, 0)),
    )                                                        # (1, 256) f32
    lmax = tf.reduce(logits, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
    e = tf.exp(logits - lmax)
    probs = e / tf.reduce(e, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    top_p, eids = tf.topk(probs, k=TOP_K, axis=-1)           # (1,8) f32 / i64
    gweights = top_p / tf.reduce(top_p, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    gweights = tf.cast(gweights, dtype="bf16")

    # ── 8 selected experts (M:737): silu(gate)*up -> down, weighted sum ──────
    # packed_* is [257,...]; eids in [0,256) only ever select routed slots,
    # same semantics as a plain [256,...] table.
    g_up = tf.gather(packed_gate_up, eids, axis=0)           # (1,8,1024,2048) bf16
    down = tf.gather(packed_down, eids, axis=0)              # (1,8,2048,512)  bf16
    tok = tf.reshape(tf.cast(ht, dtype="bf16"), new_shape=(1, 1, HIDDEN, 1))
    fused = tf.matmul(g_up, tok)                             # (1,8,1024,1)
    gate = tf.slice(fused, begin=(0, 0, 0, 0), end=(1, TOP_K, MOE_INTER, 1), strides=(1, 1, 1, 1))
    up = tf.slice(fused, begin=(0, 0, MOE_INTER, 0), end=(1, TOP_K, 2 * MOE_INTER, 1), strides=(1, 1, 1, 1))
    act = gate * tf.sigmoid(gate) * up                       # silu(gate)*up, (1,8,512,1)
    per_expert = tf.reshape(tf.matmul(down, act), new_shape=(1, TOP_K, HIDDEN))
    weighted = per_expert * tf.reshape(gweights, new_shape=(1, TOP_K, 1))
    routed = tf.reduce(weighted, axes=(1,), keepdim=False, kind=ReduceKind.SUM)  # (1, 2048)

    # ── shared expert x scalar sigmoid gate (M:807-813) ──────────────────────
    # packed slot N_EXPERTS: gate_up rows [0,512)=gate, [512,1024)=up (matches
    # moe_convert's concat(shared_gate, shared_up, axis=0) order).
    s_gu = tf.slice(
        packed_gate_up, begin=(N_EXPERTS, 0, 0),
        end=(PACKED_EXPERTS, 2 * MOE_INTER, HIDDEN), strides=(1, 1, 1),
    )
    s_gu = tf.reshape(s_gu, new_shape=(2 * MOE_INTER, HIDDEN))
    shared_gate = tf.slice(s_gu, begin=(0, 0), end=(MOE_INTER, HIDDEN), strides=(1, 1))
    shared_up = tf.slice(s_gu, begin=(MOE_INTER, 0), end=(2 * MOE_INTER, HIDDEN), strides=(1, 1))
    s_down = tf.slice(
        packed_down, begin=(N_EXPERTS, 0, 0),
        end=(PACKED_EXPERTS, HIDDEN, MOE_INTER), strides=(1, 1, 1),
    )
    shared_down = tf.reshape(s_down, new_shape=(HIDDEN, MOE_INTER))

    hb = tf.cast(ht, dtype="bf16")
    sg = tf.matmul(hb, tf.transpose(shared_gate, perm=(1, 0)))
    su = tf.matmul(hb, tf.transpose(shared_up, perm=(1, 0)))
    s_act = sg * tf.sigmoid(sg) * su                          # (1, 512)
    s_out = tf.matmul(s_act, tf.transpose(shared_down, perm=(1, 0)))          # (1, 2048)
    gate_scalar = tf.sigmoid(tf.matmul(hb, tf.transpose(shared_gate_w, perm=(1, 0))))  # (1,1)
    s_gated = s_out * gate_scalar

    # ── residual (M:874) ──────────────────────────────────────────────────
    y = tf.reshape(routed + s_gated, new_shape=(1, 1, HIDDEN))
    return x + y


@func
def moe_convert(
    post_norm: ConstTensor[(HIDDEN,), "f32"],                          # RAW ckpt layernorm weight (not +1)
    gate_w: ConstTensor[(N_EXPERTS, HIDDEN), "bf16"],
    experts_gate_up: ConstTensor[(N_EXPERTS, 2 * MOE_INTER, HIDDEN), "bf16"],
    experts_down: ConstTensor[(N_EXPERTS, HIDDEN, MOE_INTER), "bf16"],
    shared_gate: ConstTensor[(MOE_INTER, HIDDEN), "bf16"],
    shared_up: ConstTensor[(MOE_INTER, HIDDEN), "bf16"],
    shared_down: ConstTensor[(HIDDEN, MOE_INTER), "bf16"],
    shared_gate_w: ConstTensor[(1, HIDDEN), "bf16"],
):
    # Returns (post_norm_raw, router_w, packed_gate_up, packed_down,
    # shared_gate_w): the CANONICAL (kernel packed layout) weights for
    # moe_mix. Sole conversion entry point; evaluator (via moe_mix) and the
    # kernel override read this same product. The shared expert is packed
    # into slot N_EXPERTS of packed_*, gate_up ordered
    # concat(shared_gate, shared_up). Pure repack, no numeric change.
    shared_gate_up = tf.concat(shared_gate, shared_up, axis=0)             # (1024, 2048)
    shared_gate_up = tf.reshape(shared_gate_up, new_shape=(1, 2 * MOE_INTER, HIDDEN))
    packed_gate_up = tf.concat(experts_gate_up, shared_gate_up, axis=0)    # (257, 1024, 2048)

    shared_down_e = tf.reshape(shared_down, new_shape=(1, HIDDEN, MOE_INTER))
    packed_down = tf.concat(experts_down, shared_down_e, axis=0)          # (257, 2048, 512)

    post_norm_raw = tf.cast(post_norm, dtype="f32")
    router_w = tf.cast(gate_w, dtype="bf16")
    return post_norm_raw, router_w, packed_gate_up, packed_down, tf.cast(shared_gate_w, dtype="bf16")


moe_module = Module(name="moe", functions=(moe_mix, moe_convert), entry="moe_mix")
