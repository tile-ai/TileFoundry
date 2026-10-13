"""Scheduled HIR for the two ends of the prefill: model.py's `embed` and
`final`.

One Module per sequence length; `schedule finalize` lowers each kernel @func
to tir/<kernel>_{n}.py, and kernels/elementwise.py (`embed_kernel`) and
kernels/attention.py (`final_kernel`) implement them.

embed   M / 4 CTAs x 256 threads; CTA b copies rows 4 b .. 4 b + 3. Per row
        the token id is read and the table row is moved as (5, 256, 4):
        thread t moves the 8-byte vectors at columns (k * 256 + t) * 4.

final   V / 32 CTAs x 256 threads. Every CTA first normalises row `last`
        (the same (5, 256, 4) ownership as scheduled/norm.py; the row is in
        L2 after the first CTA) and stages it in shared memory as bf16. Then
        warp w of CTA b takes vocabulary rows 32 b + 4 w .. + 3: lane l reads
        the 16-byte vectors l, l + 32, ... of each row (20 per lane), the
        products are summed per lane in f32, then over the 32 lanes, and the
        sum lands in bf16 once.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
H, V, EPS = 5120, 248320, 1e-6
KV, TV, VEC = 5, 256, 4        # a row as (5, 256, 4) over 256 threads
RE = 4                          # embed: rows per CTA
NW, LANES, RW, WV = 8, 32, 4, 8  # final: warps, lanes, rows per warp, values per vector
PV = H // (LANES * WV)          # final: vectors per lane per row (20)
RB = NW * RW                    # final: vocabulary rows per CTA (32)


def embed(M: int):
    @module(entry="entry", target=_H200, topologies=(Topology("cta", M // RE), Topology("thread", TV)))
    class Embed:
        @func
        def embed(table: ConstTensor[(V, H), "bf16"], token_ids: Tensor[(M,), "i64"]):
            with Mesh(("cta",), layout=(M // RE,), names=("b",)) as cta:
                with Mesh(("thread",), layout=(TV,), names=("t",)) as th:
                    out = tf.zeros(Tensor[(M, KV, TV, VEC), "bf16"])
                    t4 = tf.reshape(table, (V, KV, TV, VEC))
                    for i in range(RE):
                        m = cta.b * RE + i
                        tok = token_ids[m]
                        row = tf.schedule((t4[tok:tok + 1, :, :, :],), op=T.copy(rmem_layout=(1, KV, TV @ th.t, VEC)))
                        out = tf.insert_slice(out, row, (m, 0, 0, 0))
                    return out

        @func
        def entry(table: ConstTensor[(V, H), "bf16"], token_ids: Tensor[(M,), "i64"]) -> Tensor[(1, M, H), "bf16"]:
            return tf.reshape(embed(table, token_ids), (1, M, H))  # noqa: F821

    Embed.__doc__ = f"embed at seq_len={M}: {M // RE} CTAs x {TV} threads"
    return Embed


def final(M: int):
    NCTA = V // RB

    @module(entry="entry", target=_H200, topologies=(Topology("cta", NCTA), Topology("thread", NW * LANES)))
    class Final:
        @func
        def final(hidden: Tensor[(1, M, H), "bf16"], last: Tensor[(1,), "i64"],
                  gamma_final: ConstTensor[(H,), "bf16"], w_head: ConstTensor[(V, H), "bf16"]):
            with Mesh(("cta",), layout=(NCTA,), names=("b",)) as cta:
                # 1. the closing norm of row `last`, every CTA, into smem
                with Mesh(("thread",), layout=(TV,), names=("t",)) as th:
                    r = last[0]
                    h4 = tf.reshape(hidden, (M, KV, TV, VEC))
                    x = tf.schedule((h4[r:r + 1, :, :, :],), op=T.copy(rmem_layout=(1, KV, TV @ th.t, VEC)))
                    g = tf.schedule((tf.reshape(gamma_final, (1, KV, TV, VEC)),), op=T.copy(rmem_layout=(1, KV, TV @ th.t, VEC)))
                    x32 = tf.cast(x, "f32")
                    sq = tf.schedule((tf.square(x32),), op=T.copy(rmem_layout=((1, KV, TV @ th.t, VEC), (0, VEC, 0, 1))))
                    ms = tf.schedule((tf.reduce(sq, (1, 2, 3), True, ReduceKind.MEAN),), op=T.copy(rmem_layout=(1, 1, 1, 1)))
                    xn = tf.cast(x32 * tf.rsqrt(ms + EPS) * (tf.cast(g, "f32") + 1.0), "bf16")
                    xs = tf.schedule((xn,), op=T.copy(smem_layout=Layout((1, KV, TV, VEC), (H, TV * VEC, VEC, 1))))
                # 2. the head rows: warp w, rows 32 b + 4 w + q; lane l, vectors l + 32 p
                with Mesh(("thread",), layout=(NW, LANES), names=("w", "l")) as th2:
                    n0 = cta.b * RB
                    w5 = tf.reshape(w_head, (V // RW, RW, PV, LANES, WV))
                    wr = tf.schedule((w5[n0 // RW:n0 // RW + NW, :, :, :, :],), op=T.copy(rmem_layout=(NW @ th2.w, RW, PV, LANES @ th2.l, WV)))
                    xv = tf.schedule((tf.reshape(xs, (1, 1, PV, LANES, WV)),), op=T.copy(rmem_layout=(1, 1, PV, LANES @ th2.l, WV)))
                    prod = tf.schedule((tf.cast(wr, "f32") * tf.cast(xv, "f32"),),
                                       op=T.copy(rmem_layout=((NW @ th2.w, RW, PV, LANES @ th2.l, WV), (0, PV * WV, WV, 0, 1))))
                    acc = tf.schedule((tf.reduce(prod, (2, 3, 4), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(NW @ th2.w, RW, 1, 1, 1)))
                    logit = tf.cast(acc, "bf16")
                    out = tf.zeros(Tensor[(V // RW, RW, 1, 1, 1), "bf16"])
                    return tf.insert_slice(out, logit, (n0 // RW, 0, 0, 0, 0))

        @func
        def entry(hidden: Tensor[(1, M, H), "bf16"], last: Tensor[(1,), "i64"],
                  gamma_final: ConstTensor[(H,), "bf16"], w_head: ConstTensor[(V, H), "bf16"]) -> Tensor[(1, V), "bf16"]:
            return tf.reshape(final(hidden, last, gamma_final, w_head), (1, V))  # noqa: F821

    Final.__doc__ = f"final at seq_len={M}: {NCTA} CTAs x {NW * LANES} threads"
    return Final


for _M in (1024, 4096, 8192):
    globals()[f"Embed{_M}"] = embed(_M)
    globals()[f"Final{_M}"] = final(_M)
