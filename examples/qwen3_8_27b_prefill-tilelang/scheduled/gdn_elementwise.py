"""Scheduled HIR for GatedDeltaNet's row-wise functions: model.py's `prep`
(causal conv + SiLU + q/k l2norm + the gates) and `gated_norm`
(Qwen3_5RMSNormGated).

One Module per sequence length; `schedule finalize` lowers each kernel @func
to tir/<kernel>_{n}.py, and kernels/elementwise.py implements it.

prep        grid (M / 64, 80) x 128 threads as (8, 16). CTA (bt, cb) takes
            rows 64 bt .. 64 bt + 63 and channel block cb (128 channels: q
            heads 0-15, k heads 16-31, v heads 32-79). Model.py's prep returns
            five tensors and scheduled lowering takes one result per
            function, so the entry calls five device functions in the one
            invocation (hir.md, "Kernel invocation"): conv_q, conv_k,
            conv_v on the CTAs whose cb falls in their block range (a MeshCoord
            range guard), gate_g and gate_beta on the cb = 0 CTAs.
  conv      the 64 rows and the 3 before them (zero for the first block) are
            staged in smem as a (67, 128) slab; tap 3 - s reads rows
            [3 - s, 67 - s): thread (s, g) owns rows 8 s .. 8 s + 7 and the
            16-byte channel vector g. f32 sum in model.py's tap order, one
            bf16 landing, SiLU in f32 landing in bf16.
  l2norm    q and k only, every step landing in bf16 (FLA's l2norm at the
            input dtype); the row's sum over the 16 lanes (`T.reduce`; the
            runtime does a 16-lane butterfly).
  gates     beta = bf16(1 / (1 + exp(-b))); g = -exp(A_log) softplus(a +
            dt_bias) in f32, softplus(x) = x for x > 20, else log1p(exp(x)),
            log1p(u) = log(1 + u) u / ((1 + u) - 1), or u where 1 + u == 1
            (no log1p instruction is registered). Thread (s, g) owns rows 8 s .. 8 s + 7, heads
            3 g .. 3 g + 2.

gated_norm  rows are (token, value head), 128 values each.
  grid      M * 48 / 16 CTAs; 256 threads as (16 rows, 16 lanes). CTA b
            takes 16 heads of one token: token b // 3, heads (b % 3) * 16 ...
  ownership lane l holds the 16-byte vector at columns l * 8 .. l * 8 + 7 of
            its row, of core, of z and of gamma.
  z         read in place from in_proj's packed output: row stride 16640,
            columns 10240 ..; the parameter's layout states that stride.
  reduce    mean of squares over the row's 16 lanes (`T.reduce`; the runtime
            does a 16-lane butterfly).
  math      model.py's order: n = bf16(x rsqrt(ms + eps)); bf16(n gamma);
            times silu(f32 z) = z / (1 + exp(-z)) in f32; one landing.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
HV, DV, VAL, EPS = 48, 128, 6144, 1e-6
P_W, Z0 = 16640, 10240        # in_proj's packed output: row stride, z's first column
RB, LN, VEC = 16, 16, 8       # rows per CTA, lanes per row, values per lane
HK, DK, CONV, KS = 16, 128, 10240, 4
B0, A0 = 16384, 16432         # b's and a's first column in in_proj's packed output
BT, SR = 64, 8                # prep: rows per CTA, rows per thread
NB = CONV // DK               # prep: 80 channel blocks
L2_EPS = 1e-6


def prep(M: int):
    NBT = M // BT

    @module(entry="entry", target=_H200, topologies=(Topology("cta", NBT * NB), Topology("thread", BT // SR * LN)))
    class Prep:
        @func
        def conv_q(qkv: Tensor[(1, M, CONV), "bf16", ((1, M, CONV), (M * P_W, P_W, 1))], conv_w: ConstTensor[(CONV, 1, KS), "bf16"]):
            with Mesh(("cta",), layout=(NBT, NB), names=("bt", "cb")) as cta:
                with Mesh(("thread",), layout=(BT // SR, LN), names=("s", "g")) as th:
                    q2 = tf.reshape(qkv, (M, CONV))
                    w2 = tf.reshape(conv_w, (CONV, KS))
                    out = tf.zeros(Tensor[(M // SR, SR, HK, LN, VEC), "bf16"])
                    m0 = cta.bt * BT
                    for h in range((cta.cb + NB - 0) % NB, HK, NB):
                        ch0 = (0 + h) * DK
                        # the 3 rows before the block (zero for the first block), then its 64 rows, in smem
                        hs = (m0 + M - (KS - 1)) % M   # the first block reads rows M-3.., masked below
                        keep = tf.cast((cta.bt + NBT - 1) // NBT, "bf16")   # 0 for the first block, else 1
                        hist = tf.schedule((tf.reshape(q2[hs:hs + KS - 1, ch0:ch0 + DK], (KS - 1, LN, VEC)),),
                                           op=T.copy(rmem_layout=(KS - 1, LN @ th.g, VEC)))
                        slab = tf.zeros(Tensor[(BT + KS - 1, LN, VEC), "bf16", Layout((BT + KS - 1, LN, VEC), (DK, VEC, 1)), "smem"])
                        slab = tf.insert_slice(slab, hist * keep, (0, 0, 0))
                        body = tf.schedule((tf.reshape(q2[m0:m0 + BT, ch0:ch0 + DK], (BT, LN, VEC)),),
                                           op=T.copy(smem_layout=Layout((BT, LN, VEC), (DK, VEC, 1))))
                        slab = tf.insert_slice(slab, body, (KS - 1, 0, 0))
                        # tap 3 - s multiplies row t - s; f32 sum in model.py's order
                        x0 = tf.schedule((tf.reshape(slab[3:3 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w0 = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 3:4], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        x1 = tf.schedule((tf.reshape(slab[2:2 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w1 = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 2:3], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        x2 = tf.schedule((tf.reshape(slab[1:1 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w2_ = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 1:2], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        x3 = tf.schedule((tf.reshape(slab[0:0 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w3 = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 0:1], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        acc = tf.cast(x0, "f32") * tf.cast(w0, "f32")
                        acc = acc + tf.cast(x1, "f32") * tf.cast(w1, "f32")
                        acc = acc + tf.cast(x2, "f32") * tf.cast(w2_, "f32")
                        acc = acc + tf.cast(x3, "f32") * tf.cast(w3, "f32")
                        xb = tf.cast(tf.cast(acc, "bf16"), "f32")
                        y = tf.cast(xb / (tf.exp(-xb) + 1.0), "bf16")
                        y32 = tf.cast(y, "f32")
                        sq = tf.schedule((tf.cast(tf.cast(y32 * y32, "bf16"), "f32"),),
                                         op=T.copy(rmem_layout=((BT // SR @ th.s, SR, LN @ th.g, VEC), (0, VEC, 0, 1))))
                        ss = tf.schedule((tf.reduce(sq, (2, 3), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(BT // SR @ th.s, SR, 1, 1)))
                        sc = tf.cast(tf.cast(ss, "bf16"), "f32")
                        sc = tf.cast(tf.cast(sc + L2_EPS, "bf16"), "f32")
                        sc = tf.cast(tf.cast(tf.rsqrt(sc), "bf16"), "f32")
                        y = tf.cast(y32 * sc, "bf16")
                        out = tf.insert_slice(out, tf.reshape(y, (BT // SR, SR, 1, LN, VEC)), (cta.bt * (BT // SR), 0, h, 0, 0))
                    return out

        @func
        def conv_k(qkv: Tensor[(1, M, CONV), "bf16", ((1, M, CONV), (M * P_W, P_W, 1))], conv_w: ConstTensor[(CONV, 1, KS), "bf16"]):
            with Mesh(("cta",), layout=(NBT, NB), names=("bt", "cb")) as cta:
                with Mesh(("thread",), layout=(BT // SR, LN), names=("s", "g")) as th:
                    q2 = tf.reshape(qkv, (M, CONV))
                    w2 = tf.reshape(conv_w, (CONV, KS))
                    out = tf.zeros(Tensor[(M // SR, SR, HK, LN, VEC), "bf16"])
                    m0 = cta.bt * BT
                    for h in range((cta.cb + NB - HK) % NB, HK, NB):
                        ch0 = (HK + h) * DK
                        # the 3 rows before the block (zero for the first block), then its 64 rows, in smem
                        hs = (m0 + M - (KS - 1)) % M   # the first block reads rows M-3.., masked below
                        keep = tf.cast((cta.bt + NBT - 1) // NBT, "bf16")   # 0 for the first block, else 1
                        hist = tf.schedule((tf.reshape(q2[hs:hs + KS - 1, ch0:ch0 + DK], (KS - 1, LN, VEC)),),
                                           op=T.copy(rmem_layout=(KS - 1, LN @ th.g, VEC)))
                        slab = tf.zeros(Tensor[(BT + KS - 1, LN, VEC), "bf16", Layout((BT + KS - 1, LN, VEC), (DK, VEC, 1)), "smem"])
                        slab = tf.insert_slice(slab, hist * keep, (0, 0, 0))
                        body = tf.schedule((tf.reshape(q2[m0:m0 + BT, ch0:ch0 + DK], (BT, LN, VEC)),),
                                           op=T.copy(smem_layout=Layout((BT, LN, VEC), (DK, VEC, 1))))
                        slab = tf.insert_slice(slab, body, (KS - 1, 0, 0))
                        # tap 3 - s multiplies row t - s; f32 sum in model.py's order
                        x0 = tf.schedule((tf.reshape(slab[3:3 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w0 = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 3:4], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        x1 = tf.schedule((tf.reshape(slab[2:2 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w1 = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 2:3], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        x2 = tf.schedule((tf.reshape(slab[1:1 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w2_ = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 1:2], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        x3 = tf.schedule((tf.reshape(slab[0:0 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w3 = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 0:1], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        acc = tf.cast(x0, "f32") * tf.cast(w0, "f32")
                        acc = acc + tf.cast(x1, "f32") * tf.cast(w1, "f32")
                        acc = acc + tf.cast(x2, "f32") * tf.cast(w2_, "f32")
                        acc = acc + tf.cast(x3, "f32") * tf.cast(w3, "f32")
                        xb = tf.cast(tf.cast(acc, "bf16"), "f32")
                        y = tf.cast(xb / (tf.exp(-xb) + 1.0), "bf16")
                        y32 = tf.cast(y, "f32")
                        sq = tf.schedule((tf.cast(tf.cast(y32 * y32, "bf16"), "f32"),),
                                         op=T.copy(rmem_layout=((BT // SR @ th.s, SR, LN @ th.g, VEC), (0, VEC, 0, 1))))
                        ss = tf.schedule((tf.reduce(sq, (2, 3), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(BT // SR @ th.s, SR, 1, 1)))
                        sc = tf.cast(tf.cast(ss, "bf16"), "f32")
                        sc = tf.cast(tf.cast(sc + L2_EPS, "bf16"), "f32")
                        sc = tf.cast(tf.cast(tf.rsqrt(sc), "bf16"), "f32")
                        y = tf.cast(y32 * sc, "bf16")
                        out = tf.insert_slice(out, tf.reshape(y, (BT // SR, SR, 1, LN, VEC)), (cta.bt * (BT // SR), 0, h, 0, 0))
                    return out

        @func
        def conv_v(qkv: Tensor[(1, M, CONV), "bf16", ((1, M, CONV), (M * P_W, P_W, 1))], conv_w: ConstTensor[(CONV, 1, KS), "bf16"]):
            with Mesh(("cta",), layout=(NBT, NB), names=("bt", "cb")) as cta:
                with Mesh(("thread",), layout=(BT // SR, LN), names=("s", "g")) as th:
                    q2 = tf.reshape(qkv, (M, CONV))
                    w2 = tf.reshape(conv_w, (CONV, KS))
                    out = tf.zeros(Tensor[(M // SR, SR, HV, LN, VEC), "bf16"])
                    m0 = cta.bt * BT
                    for h in range((cta.cb + NB - 2 * HK) % NB, HV, NB):
                        ch0 = (2 * HK + h) * DK
                        # the 3 rows before the block (zero for the first block), then its 64 rows, in smem
                        hs = (m0 + M - (KS - 1)) % M   # the first block reads rows M-3.., masked below
                        keep = tf.cast((cta.bt + NBT - 1) // NBT, "bf16")   # 0 for the first block, else 1
                        hist = tf.schedule((tf.reshape(q2[hs:hs + KS - 1, ch0:ch0 + DK], (KS - 1, LN, VEC)),),
                                           op=T.copy(rmem_layout=(KS - 1, LN @ th.g, VEC)))
                        slab = tf.zeros(Tensor[(BT + KS - 1, LN, VEC), "bf16", Layout((BT + KS - 1, LN, VEC), (DK, VEC, 1)), "smem"])
                        slab = tf.insert_slice(slab, hist * keep, (0, 0, 0))
                        body = tf.schedule((tf.reshape(q2[m0:m0 + BT, ch0:ch0 + DK], (BT, LN, VEC)),),
                                           op=T.copy(smem_layout=Layout((BT, LN, VEC), (DK, VEC, 1))))
                        slab = tf.insert_slice(slab, body, (KS - 1, 0, 0))
                        # tap 3 - s multiplies row t - s; f32 sum in model.py's order
                        x0 = tf.schedule((tf.reshape(slab[3:3 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w0 = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 3:4], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        x1 = tf.schedule((tf.reshape(slab[2:2 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w1 = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 2:3], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        x2 = tf.schedule((tf.reshape(slab[1:1 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w2_ = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 1:2], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        x3 = tf.schedule((tf.reshape(slab[0:0 + BT, :, :], (BT // SR, SR, LN, VEC)),),
                                            op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, VEC)))
                        w3 = tf.schedule((tf.reshape(w2[ch0:ch0 + DK, 0:1], (1, 1, LN, VEC)),),
                                            op=T.copy(rmem_layout=(1, 1, LN @ th.g, VEC)))
                        acc = tf.cast(x0, "f32") * tf.cast(w0, "f32")
                        acc = acc + tf.cast(x1, "f32") * tf.cast(w1, "f32")
                        acc = acc + tf.cast(x2, "f32") * tf.cast(w2_, "f32")
                        acc = acc + tf.cast(x3, "f32") * tf.cast(w3, "f32")
                        xb = tf.cast(tf.cast(acc, "bf16"), "f32")
                        y = tf.cast(xb / (tf.exp(-xb) + 1.0), "bf16")
                        out = tf.insert_slice(out, tf.reshape(y, (BT // SR, SR, 1, LN, VEC)), (cta.bt * (BT // SR), 0, h, 0, 0))
                    return out

        @func
        def gate_beta(b: Tensor[(1, M, HV), "bf16", ((1, M, HV), (M * P_W, P_W, 1))]):
            with Mesh(("cta",), layout=(NBT, NB), names=("bt", "cb")) as cta:
                with Mesh(("thread",), layout=(BT // SR, LN), names=("s", "g")) as th:
                    b3 = tf.reshape(b, (M // SR, SR, LN, HV // LN))
                    out = tf.zeros(Tensor[(M // SR, SR, LN, HV // LN), "bf16"])
                    for _ in range(cta.cb, 1, NB):
                        r0 = cta.bt * (BT // SR)
                        bb = tf.cast(tf.schedule((b3[r0:r0 + BT // SR, :, :, :],), op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, HV // LN))), "f32")
                        out = tf.insert_slice(out, tf.cast(1.0 / (tf.exp(-bb) + 1.0), "bf16"), (r0, 0, 0, 0))
                    return out

        @func
        def gate_g(a: Tensor[(1, M, HV), "bf16", ((1, M, HV), (M * P_W, P_W, 1))], a_log: ConstTensor[(HV,), "bf16"],
                   dt_bias: ConstTensor[(HV,), "bf16"]):
            with Mesh(("cta",), layout=(NBT, NB), names=("bt", "cb")) as cta:
                with Mesh(("thread",), layout=(BT // SR, LN), names=("s", "g")) as th:
                    a3 = tf.reshape(a, (M // SR, SR, LN, HV // LN))
                    out = tf.zeros(Tensor[(M // SR, SR, LN, HV // LN), "f32"])
                    for _ in range(cta.cb, 1, NB):
                        r0 = cta.bt * (BT // SR)
                        aa = tf.cast(tf.schedule((a3[r0:r0 + BT // SR, :, :, :],), op=T.copy(rmem_layout=(BT // SR @ th.s, SR, LN @ th.g, HV // LN))), "f32")
                        al = tf.cast(tf.schedule((tf.reshape(a_log, (1, 1, LN, HV // LN)),), op=T.copy(rmem_layout=(1, 1, LN @ th.g, HV // LN))), "f32")
                        db = tf.cast(tf.schedule((tf.reshape(dt_bias, (1, 1, LN, HV // LN)),), op=T.copy(rmem_layout=(1, 1, LN @ th.g, HV // LN))), "f32")
                        x = aa + db
                        u = tf.exp(x)
                        v1 = u + 1.0
                        lp = tf.where(v1 == 1.0, u, tf.log(v1) * u / (v1 - 1.0))   # log1p(u)
                        sp = tf.where(x > 20.0, x, lp)
                        out = tf.insert_slice(out, -tf.exp(al) * sp, (r0, 0, 0, 0))
                    return out

        @func
        def entry(qkv: Tensor[(1, M, CONV), "bf16", ((1, M, CONV), (M * P_W, P_W, 1))], b: Tensor[(1, M, HV), "bf16", ((1, M, HV), (M * P_W, P_W, 1))],
                  a: Tensor[(1, M, HV), "bf16", ((1, M, HV), (M * P_W, P_W, 1))], conv_w: ConstTensor[(CONV, 1, KS), "bf16"],
                  a_log: ConstTensor[(HV,), "bf16"], dt_bias: ConstTensor[(HV,), "bf16"]):
            q = tf.reshape(conv_q(qkv, conv_w), (1, M, HK, DK))  # noqa: F821
            k = tf.reshape(conv_k(qkv, conv_w), (1, M, HK, DK))  # noqa: F821
            v = tf.reshape(conv_v(qkv, conv_w), (1, M, HV, DV))  # noqa: F821
            g = tf.reshape(gate_g(a, a_log, dt_bias), (1, M, HV))  # noqa: F821
            beta = tf.reshape(gate_beta(b), (1, M, HV))  # noqa: F821
            return q, k, v, g, beta

    Prep.__doc__ = f"prep at seq_len={M}: ({NBT}, {NB}) CTAs x {BT // SR * LN} threads"
    return Prep




def gated_norm(M: int):
    NCTA = M * HV // RB
    HB = HV // RB             # CTAs per token

    @module(entry="entry", target=_H200, topologies=(Topology("cta", NCTA), Topology("thread", RB * LN)))
    class GatedNorm:
        @func
        def gated_norm(core: Tensor[(1, M, HV, DV), "bf16"],
                       z: Tensor[(1, M, VAL), "bf16", ((1, M, VAL), (M * P_W, P_W, 1))],
                       gamma_gdn: ConstTensor[(DV,), "bf16"]):
            with Mesh(("cta",), layout=(NCTA,), names=("b",)) as cta:
                with Mesh(("thread",), layout=(RB, LN), names=("row", "lane")) as th:
                    m = cta.b // HB
                    h0 = (cta.b % HB) * RB
                    c4 = tf.reshape(core, (M, HV, LN, VEC))
                    z4 = tf.reshape(z, (M, HV, LN, VEC))
                    x = tf.schedule((c4[m:m + 1, h0:h0 + RB, :, :],), op=T.copy(rmem_layout=(1, RB @ th.row, LN @ th.lane, VEC)))
                    zz = tf.schedule((z4[m:m + 1, h0:h0 + RB, :, :],), op=T.copy(rmem_layout=(1, RB @ th.row, LN @ th.lane, VEC)))
                    g = tf.schedule((tf.reshape(gamma_gdn, (1, 1, LN, VEC)),), op=T.copy(rmem_layout=(1, 1, LN @ th.lane, VEC)))
                    x32 = tf.cast(x, "f32")
                    sq = tf.schedule((tf.square(x32),), op=T.copy(rmem_layout=((1, RB @ th.row, LN @ th.lane, VEC), (0, 0, 0, 1))))
                    ms = tf.schedule((tf.reduce(sq, (2, 3), True, ReduceKind.MEAN),),
                                     op=T.copy(rmem_layout=(1, RB @ th.row, 1, 1)))
                    n = tf.cast(x32 * tf.rsqrt(ms + EPS), "bf16")
                    nw = tf.cast(tf.cast(n, "f32") * tf.cast(g, "f32"), "bf16")
                    z32 = tf.cast(zz, "f32")
                    gate = z32 / (tf.exp(-z32) + 1.0)
                    y = tf.cast(tf.cast(nw, "f32") * gate, "bf16")
                    return tf.insert_slice(tf.zeros(Tensor[(M, HV, LN, VEC), "bf16"]), y, (m, h0, 0, 0))

        @func
        def entry(core: Tensor[(1, M, HV, DV), "bf16"],
                  z: Tensor[(1, M, VAL), "bf16", ((1, M, VAL), (M * P_W, P_W, 1))],
                  gamma_gdn: ConstTensor[(DV,), "bf16"]) -> Tensor[(1, M, VAL), "bf16"]:
            return tf.reshape(gated_norm(core, z, gamma_gdn), (1, M, VAL))  # noqa: F821

    GatedNorm.__doc__ = f"gated_norm at seq_len={M}: {NCTA} CTAs x {RB * LN} threads"
    return GatedNorm


for _M in (1024, 4096, 8192):
    globals()[f"Prep{_M}"] = prep(_M)
    globals()[f"GatedNorm{_M}"] = gated_norm(_M)
