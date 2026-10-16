"""Scheduled HIR: attention prep at prompt length 256.

Implements model.py Qwen3_8AttentionLayer.gated_attention up to the core:
  q = partial_rope(rms_norm(qg[..., :256], 1 + gamma_q)), k = partial_rope_kv(rms_norm(k, 1 + gamma_k)), v.
Runtime kernel: runtime/kernels/attention.py make_prep(T) (`attn_prep`).

Decomposition (as the kernel): one CTA per token, 1024 threads = 32 warps; warp w < 24
handles query head w, 24 <= w < 28 key head w - 24, w >= 28 copies value head w - 28.
Each lane holds elements 4l..4l+3 and 128+4l..128+4l+3 of its head row; the RMS
statistic is one cross-lane reduce; the row is landed once in bf16.
RoPE (rotate-half on dims 0..63, bf16 arithmetic): the pair (d, d+32) must meet in one
lane. TileFoundry's H200 instruction set has no intra-warp shuffle (schedule facts:
copy, copy_async, copy_async_tensor, ldmatrix, tiled_mma, binary/unary/cast/where/
clamp/relu/reduce), so this program states the exchange through shared memory: the
normalized row is staged in smem and lane j reads the pair (j, j+32). The kernel
realizes the same pairing in registers: the pair sits in lanes l and l^8 and is
exchanged with shfl_xor(., 8). Values and roundings are identical (tested bitwise).
Result: scheduled lowering accepts one tensor result, so q, k and v are stated as one
(T, 32, 256) tensor whose head slots 0..23 / 24..27 / 28..31 are q / k / v -- the
kernel's warp-slot order; the kernel writes the same rows into three buffers.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

T_ = 256
D = 256
ROT = 64
HALF = 32
EPS = 1e-6
QW = 24 * 512


@module(entry="attn_prep", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", T_), Topology("thread", 1024)))
class SCHED_ATTN_PREP_T256:
    @func
    def attn_prep(qkv: Tensor[(T_, 14336), "bf16"], gamma_q: Tensor[(D,), "bf16"], gamma_k: Tensor[(D,), "bf16"],
                  cos: Tensor[(T_, ROT), "bf16"], sin: Tensor[(T_, ROT), "bf16"]):
        with Mesh(("cta",), layout=(T_,), names=("t",)) as cta:
            with Mesh(("thread",), layout=(32, 32), names=("warp", "lane")) as th:
                out = tf.zeros(Tensor[(T_, 32, D), "bf16"])
                srcq = tf.reshape(qkv[cta.t:cta.t + 1, 0:QW], new_shape=(24, 512))[:, 0:D]
                with Mesh(th[0:24, :], layout=(24, 32), names=("head", "lane")) as mq:
                    x = tf.schedule((srcq, tf.empty(Tensor[(24, D), "bf16", ((24 @ mq.head, 2, 32 @ mq.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    g = tf.schedule((tf.reshape(gamma_q, new_shape=(1, D)), tf.empty(Tensor[(1, D), "bf16", ((1, 2, 32 @ mq.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    xf = tf.cast(x, dtype="f32")
                    ms = tf.reduce(xf * xf, axes=(-1,), keepdim=True, kind="mean")
                    n = tf.cast(xf * tf.rsqrt(ms + EPS) * (tf.cast(g, dtype="f32") + 1.0), dtype="bf16")
                    ns = tf.schedule((n, tf.empty(Tensor[(24, D), "bf16", (24, D), "smem"])), op=T.copy())
                    lo = tf.cast(tf.schedule((ns[:, 0:HALF], tf.empty(Tensor[(24, HALF), "bf16", ((24 @ mq.head, 32 @ mq.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    hi = tf.cast(tf.schedule((ns[:, HALF:ROT], tf.empty(Tensor[(24, HALF), "bf16", ((24 @ mq.head, 32 @ mq.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    c_lo = tf.cast(tf.schedule((cos[cta.t:cta.t + 1, 0:HALF], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ mq.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    c_hi = tf.cast(tf.schedule((cos[cta.t:cta.t + 1, HALF:ROT], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ mq.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    s_lo = tf.cast(tf.schedule((sin[cta.t:cta.t + 1, 0:HALF], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ mq.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    s_hi = tf.cast(tf.schedule((sin[cta.t:cta.t + 1, HALF:ROT], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ mq.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    r_lo = tf.cast(tf.cast(tf.cast(lo * c_lo, dtype="bf16"), dtype="f32") + tf.cast(tf.cast(tf.neg(hi) * s_lo, dtype="bf16"), dtype="f32"), dtype="bf16")
                    r_hi = tf.cast(tf.cast(tf.cast(hi * c_hi, dtype="bf16"), dtype="f32") + tf.cast(tf.cast(lo * s_hi, dtype="bf16"), dtype="f32"), dtype="bf16")
                    o_lo = tf.schedule((tf.reshape(r_lo, new_shape=(1, 24, HALF)), tf.empty(Tensor[(1, 24, HALF), "bf16", (1, 24, HALF)])), op=T.copy())
                    o_hi = tf.schedule((tf.reshape(r_hi, new_shape=(1, 24, HALF)), tf.empty(Tensor[(1, 24, HALF), "bf16", (1, 24, HALF)])), op=T.copy())
                    o_pass = tf.schedule((tf.reshape(ns[:, ROT:D], new_shape=(1, 24, D - ROT)), tf.empty(Tensor[(1, 24, D - ROT), "bf16", (1, 24, D - ROT)])), op=T.copy())
                    out = tf.insert_slice(out, o_lo, (cta.t, 0, 0))
                    out = tf.insert_slice(out, o_hi, (cta.t, 0, HALF))
                    out = tf.insert_slice(out, o_pass, (cta.t, 0, ROT))
                srck = tf.reshape(qkv[cta.t:cta.t + 1, QW:QW + 1024], new_shape=(4, D))
                with Mesh(th[24:28, :], layout=(4, 32), names=("head", "lane")) as mk:
                    x = tf.schedule((srck, tf.empty(Tensor[(4, D), "bf16", ((4 @ mk.head, 2, 32 @ mk.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    g = tf.schedule((tf.reshape(gamma_k, new_shape=(1, D)), tf.empty(Tensor[(1, D), "bf16", ((1, 2, 32 @ mk.lane, 4), (0, 4, 0, 1)), "rmem"])), op=T.copy())
                    xf = tf.cast(x, dtype="f32")
                    ms = tf.reduce(xf * xf, axes=(-1,), keepdim=True, kind="mean")
                    n = tf.cast(xf * tf.rsqrt(ms + EPS) * (tf.cast(g, dtype="f32") + 1.0), dtype="bf16")
                    ns = tf.schedule((n, tf.empty(Tensor[(4, D), "bf16", (4, D), "smem"])), op=T.copy())
                    lo = tf.cast(tf.schedule((ns[:, 0:HALF], tf.empty(Tensor[(4, HALF), "bf16", ((4 @ mk.head, 32 @ mk.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    hi = tf.cast(tf.schedule((ns[:, HALF:ROT], tf.empty(Tensor[(4, HALF), "bf16", ((4 @ mk.head, 32 @ mk.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    c_lo = tf.cast(tf.schedule((cos[cta.t:cta.t + 1, 0:HALF], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ mk.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    c_hi = tf.cast(tf.schedule((cos[cta.t:cta.t + 1, HALF:ROT], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ mk.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    s_lo = tf.cast(tf.schedule((sin[cta.t:cta.t + 1, 0:HALF], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ mk.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    s_hi = tf.cast(tf.schedule((sin[cta.t:cta.t + 1, HALF:ROT], tf.empty(Tensor[(1, HALF), "bf16", ((1, 32 @ mk.lane), (0, 0)), "rmem"])), op=T.copy()), dtype="f32")
                    r_lo = tf.cast(tf.cast(tf.cast(lo * c_lo, dtype="bf16"), dtype="f32") + tf.cast(tf.cast(tf.neg(hi) * s_lo, dtype="bf16"), dtype="f32"), dtype="bf16")
                    r_hi = tf.cast(tf.cast(tf.cast(hi * c_hi, dtype="bf16"), dtype="f32") + tf.cast(tf.cast(lo * s_hi, dtype="bf16"), dtype="f32"), dtype="bf16")
                    o_lo = tf.schedule((tf.reshape(r_lo, new_shape=(1, 4, HALF)), tf.empty(Tensor[(1, 4, HALF), "bf16", (1, 4, HALF)])), op=T.copy())
                    o_hi = tf.schedule((tf.reshape(r_hi, new_shape=(1, 4, HALF)), tf.empty(Tensor[(1, 4, HALF), "bf16", (1, 4, HALF)])), op=T.copy())
                    o_pass = tf.schedule((tf.reshape(ns[:, ROT:D], new_shape=(1, 4, D - ROT)), tf.empty(Tensor[(1, 4, D - ROT), "bf16", (1, 4, D - ROT)])), op=T.copy())
                    out = tf.insert_slice(out, o_lo, (cta.t, 24, 0))
                    out = tf.insert_slice(out, o_hi, (cta.t, 24, HALF))
                    out = tf.insert_slice(out, o_pass, (cta.t, 24, ROT))
                with Mesh(th[28:32, :], layout=(4, 32), names=("head", "lane")) as mv:
                    vv = tf.schedule((tf.reshape(qkv[cta.t:cta.t + 1, QW + 1024:QW + 2048], new_shape=(4, D)), tf.empty(Tensor[(4, D), "bf16", ((4 @ mv.head, 32 @ mv.lane, 8), (0, 0, 1)), "rmem"])), op=T.copy())
                    vo = tf.schedule((tf.reshape(vv, new_shape=(1, 4, D)), tf.empty(Tensor[(1, 4, D), "bf16", (1, 4, D)])), op=T.copy())
                    out = tf.insert_slice(out, vo, (cta.t, 28, 0))
            return out
