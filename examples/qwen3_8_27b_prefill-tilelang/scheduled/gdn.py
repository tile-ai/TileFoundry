"""Scheduled HIR for model.py's `GatedDeltaNet.delta_rule`: the chunked gated
delta rule (64-token chunks) as two kernels.

One Module per sequence length. The entry is one invocation of three device
functions (hir.md, "Kernel invocation"); `schedule finalize` lowers each to
tir/delta_rule_{intra_g,intra_wup,recur_out}_{n}.py. kernels/gdn.py runs
intra_g and intra_wup as one launch (`intra_kernel`, they share a grid and
intra_wup's CTA reads only the gc its own intra_g CTA wrote), and recur_out as
a second launch (`recur_out_kernel`) -- the launch boundary is the grid-wide
barrier recur_out needs, since it reads every chunk.

  intra_g    grid (chunks, 48 value heads), 128 threads; the first 64 sum g
             over the chunk prefix (thread i: g_0 .. g_i in order).
  intra_wup  same grid. K, Q, V of the chunk by TMA (SW128); K K^T and Q K^T
             on the tensor cores (m64n64k16, K-major); decay exp(g_i - g_j)
             and the masks in the WGMMA fragment's ownership; P = bf16(scale
             q k^T decay) lower incl. the diagonal; L = -strict_lower(beta k
             k^T decay) in f32 smem (one padding column). T = (I - L)^-1:
             phase 1, the four 16 x 16 diagonal blocks, 16 threads each, thread
             j holding column j and rows in order; phase 2a, the two 32 x 32
             super-blocks' off-diagonal blocks, stored transposed above the
             diagonal; phase 2b, the lower-left 32 x 32 block -- f32 products
             summed per thread (no f32 / tf32 MMA atom is registered for
             H200); a block read transposed is a bitcast view of the f32
             matrix with swapped strides, read in place (no copy). Then w = T bf16(beta k e^g) and u = T bf16(beta v) as their
             transposes on the tensor cores (A = X^T MN-major, B = T^T
             K-major), staged and stored by TMA with P into one packed tensor
             [w | u | P row] per (token, head).
  recur_out  grid (2 value halves, 48 heads), 384 threads: warpgroup 2 loads
             w, k, q, u, P, gc of chunk c by TMA (2-deep); warpgroup 0 holds
             the state S (d x 64, f32) and runs v_new = u - w S and S <- e^gl S
             + K^T (v_new e^(gl - g)) with the bf16 copy of S, v_new and the
             decays in 3-deep smem rings; warpgroup 1 reads out o = scale e^g
             (q S_c) + P v_new, staged and stored by TMA.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
HK, HV, D, CH, BV = 16, 48, 128, 64, 64
NV = D // BV
VPK = HV // HK
SCALE = D ** -0.5
WUP = 2 * D + CH       # packed intra output per (token, head): w | u | P row


def delta_rule_intra(M: int):
    """Invocation 1 of model.py's delta_rule: (gc, packed w | u | P)."""
    C = M // CH

    @module(entry="entry", target=_H200, topologies=(Topology("cta", C * HV), Topology("thread", 128)))
    class DeltaRuleIntra:
        @func
        def intra_g(g: Tensor[(M, HV), "f32"], iota: Tensor[(CH,), "i32"]):
            """gc = the prefix sum of g inside each 64-token chunk: the chunk's g staged
            in smem (thread i: g_i), then thread i of the first 64 sums g_j for
            jc <= ir (j ascending)."""
            with Mesh(("cta",), layout=(C, HV), names=("c", "h")) as cta:
                with Mesh(("thread",), layout=(2, 64), names=("half", "i")) as threads:
                    out = tf.zeros(Tensor[(HV, M, 1), "f32"])
                    t0 = cta.c * CH
                    with Mesh(threads[0, :], layout=(CH,), names=("i",)) as r:
                        g1 = tf.schedule((tf.reshape(g[t0:t0 + CH, cta.h:cta.h + 1], (CH,)),), op=T.copy(rmem_layout=(CH @ r.i,)))
                        gsm = tf.schedule((g1,), op=T.copy(smem_layout=Layout((CH,), (1,))))
                        gv = tf.schedule((tf.reshape(gsm, (1, CH)),), op=T.copy(rmem_layout=(1, CH)))
                        ir = tf.schedule((tf.reshape(iota, (CH, 1)),), op=T.copy(rmem_layout=(CH @ r.i, 1)))
                        jc = tf.schedule((tf.reshape(iota, (1, CH)),), op=T.copy(rmem_layout=(1, CH)))
                        gm = tf.schedule((gv * tf.cast(jc <= ir, "f32"),), op=T.copy(rmem_layout=((CH @ r.i, CH), (0, 1))))
                        gcum = tf.schedule((tf.reduce(gm, (1,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(CH @ r.i, 1)))
                        out = tf.insert_slice(out, tf.reshape(gcum, (1, CH, 1)), (cta.h, t0, 0))
                    return out

        @func
        def intra_wup(q: Tensor[(M, HK, D), "bf16"], k: Tensor[(M, HK, D), "bf16"], v: Tensor[(M, HV, D), "bf16"],
                      gc: Tensor[(HV, M), "f32"], beta: Tensor[(M, HV), "bf16"], iota: Tensor[(CH,), "i32"]):
            with Mesh(("cta",), layout=(C, HV), names=("c", "h")) as cta:
                with Mesh(("thread",), layout=(4, 32), names=("warp", "lane")) as threads:
                    w_qk = T.cuda.sm90.Wgmma(n=CH, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                    w_tx = T.cuda.sm90.Wgmma(n=CH, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)
                    out = tf.zeros(Tensor[(M, HV * WUP), "bf16"])
                    t0 = cta.c * CH
                    kh = cta.h // VPK
                    with threads[0, :] as _loader:
                        ks = tf.schedule((tf.reshape(k[t0:t0 + CH, kh:kh + 1, :], (CH, D)),), op=T.copy_async_tensor(smem_layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3)))
                        qs = tf.schedule((tf.reshape(q[t0:t0 + CH, kh:kh + 1, :], (CH, D)),), op=T.copy_async_tensor(smem_layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3)))
                    with Mesh(threads, layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as s:
                        kt = tf.bitcast(ks, layout=Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3))
                        kk = tf.schedule((tf.zeros(Tensor[(CH, CH), "f32", ((4 @ s.warp, 2, 8 @ s.lane8, 8, 4 @ s.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"]), ks, kt),
                                         op=T.tiled_mma(atom=w_qk), repeat=(1, 1, 8))
                        qk = tf.schedule((tf.zeros(Tensor[(CH, CH), "f32", ((4 @ s.warp, 2, 8 @ s.lane8, 8, 4 @ s.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"]), qs, kt),
                                         op=T.tiled_mma(atom=w_qk), repeat=(1, 1, 8))
                    # V lands after Q K^T, in Q's place
                    with threads[0, :] as _loader2:
                        vs = tf.schedule((tf.reshape(v[t0:t0 + CH, cta.h:cta.h + 1, :], (CH, D)),), op=T.copy_async_tensor(smem_layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3)))
                    # the chunk's gc and beta, staged in smem once (thread i of the first 64: row i), as rows 0
                    # and 1 of one 1024-byte tile: the SW128 tiles placed after it then start on 1024-byte
                    # boundaries, which TMA's 128-byte swizzle needs (it swizzles by absolute address)
                    with Mesh(threads[0:2, :], layout=(CH,), names=("i",)) as r2:
                        gb = tf.zeros(Tensor[(4, CH), "f32", Layout((4, CH), (CH, 1)), "smem"])
                        g1 = tf.schedule((tf.reshape(gc[cta.h:cta.h + 1, t0:t0 + CH], (1, CH)),), op=T.copy(rmem_layout=(1, CH @ r2.i)))
                        gb = tf.insert_slice(gb, g1, (0, 0))
                        b1 = tf.cast(tf.schedule((tf.reshape(beta[t0:t0 + CH, cta.h:cta.h + 1], (1, CH)),), op=T.copy(rmem_layout=(1, CH @ r2.i))), "f32")
                        gb = tf.insert_slice(gb, b1, (1, 0))
                        gcs = gb[0:1, :]
                        bss = gb[1:2, :]
                    with Mesh(threads, layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as s:
                        ir = tf.schedule((tf.reshape(iota, (CH, 1)),), op=T.copy(rmem_layout=((8 @ s.lane8, 2, 4 @ s.warp, 1), (1, 8, 16, 0))))
                        jc = tf.schedule((tf.reshape(iota, (1, CH)),), op=T.copy(rmem_layout=((1, 2, 4 @ s.lane4, 8), (0, 64, 128, 512))))
                        dif = tf.schedule((ir - jc,), op=T.copy(rmem_layout=((4 @ s.warp, 2, 8 @ s.lane8, 8, 4 @ s.lane4, 2,), (16, 8, 1, 512, 128, 64))))
                        zi = tf.zeros(Tensor[(1, 1), "i32", (1, 1), "rmem"])   # an i32 zero (an integer literal is i64)
                        gr = tf.schedule((tf.reshape(gcs, (CH, 1)),), op=T.copy(rmem_layout=((8 @ s.lane8, 2, 4 @ s.warp, 1), (1, 8, 16, 0))))
                        gcl = tf.schedule((tf.reshape(gcs, (1, CH)),), op=T.copy(rmem_layout=((1, 2, 4 @ s.lane4, 8), (0, 64, 128, 512))))
                        br = tf.schedule((tf.reshape(bss, (CH, 1)),), op=T.copy(rmem_layout=((8 @ s.lane8, 2, 4 @ s.warp, 1), (1, 8, 16, 0))))
                        dec = tf.exp(gr - gcl)
                        # P = bf16(scale q k^T decay), lower including the diagonal
                        pm = tf.cast(tf.where(dif >= zi, qk * dec * SCALE, 0.0), "bf16")
                        ps = tf.schedule((pm,), op=T.copy(smem_layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3)))
                        # L = -strict_lower(beta_i (k_i . k_j) decay_ij), f32, one padding column
                        # (CH, CH + 1): the padding column is part of the tensor, so placement counts its bytes
                        asm = tf.zeros(Tensor[(CH, CH + 1), "f32", Layout((CH, CH + 1), (CH + 1, 1)), "smem"])
                        asm = tf.insert_slice(asm, tf.where(dif > zi, -(kk * br * dec), 0.0), (0, 0))
                    with threads[0, :] as _storer:
                        tile = tf.schedule((ps,), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (t0, cta.h * WUP + 2 * D))
                    # the X chunks, before the solve (they need K, V, beta, gc but not T): X_w = bf16(beta k e^gc),
                    # X_u = bf16(beta v), each 64 x 64 in SW128 smem, MN-major B operands of the products below
                    with Mesh(threads, layout=(8, 16), names=("ti", "tj")) as xw0:
                        bq = tf.schedule((tf.reshape(bss, (8, 8, 1, 1)),), op=T.copy(rmem_layout=(8, 8 @ xw0.ti, 1, 1)))
                        sx = tf.cast(tf.schedule((tf.reshape(ks[:, 0:64], (8, 8, 16, 4)),), op=T.copy(rmem_layout=(8, 8 @ xw0.ti, 16 @ xw0.tj, 4))), "f32")
                        gq = tf.schedule((tf.reshape(gcs, (8, 8, 1, 1)),), op=T.copy(rmem_layout=(8, 8 @ xw0.ti, 1, 1)))
                        xv = sx * bq * tf.exp(gq)
                        xw0s = tf.schedule((tf.reshape(tf.cast(xv, "bf16"), (CH, 64)),), op=T.copy(smem_layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3)))
                    with Mesh(threads, layout=(8, 16), names=("ti", "tj")) as xw1:
                        bq = tf.schedule((tf.reshape(bss, (8, 8, 1, 1)),), op=T.copy(rmem_layout=(8, 8 @ xw1.ti, 1, 1)))
                        sx = tf.cast(tf.schedule((tf.reshape(ks[:, 64:128], (8, 8, 16, 4)),), op=T.copy(rmem_layout=(8, 8 @ xw1.ti, 16 @ xw1.tj, 4))), "f32")
                        gq = tf.schedule((tf.reshape(gcs, (8, 8, 1, 1)),), op=T.copy(rmem_layout=(8, 8 @ xw1.ti, 1, 1)))
                        xv = sx * bq * tf.exp(gq)
                        xw1s = tf.schedule((tf.reshape(tf.cast(xv, "bf16"), (CH, 64)),), op=T.copy(smem_layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3)))
                    with Mesh(threads, layout=(8, 16), names=("ti", "tj")) as xu0:
                        bq = tf.schedule((tf.reshape(bss, (8, 8, 1, 1)),), op=T.copy(rmem_layout=(8, 8 @ xu0.ti, 1, 1)))
                        sx = tf.cast(tf.schedule((tf.reshape(vs[:, 0:64], (8, 8, 16, 4)),), op=T.copy(rmem_layout=(8, 8 @ xu0.ti, 16 @ xu0.tj, 4))), "f32")
                        xv = sx * bq
                        xu0s = tf.schedule((tf.reshape(tf.cast(xv, "bf16"), (CH, 64)),), op=T.copy(smem_layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3)))
                    with Mesh(threads, layout=(8, 16), names=("ti", "tj")) as xu1:
                        bq = tf.schedule((tf.reshape(bss, (8, 8, 1, 1)),), op=T.copy(rmem_layout=(8, 8 @ xu1.ti, 1, 1)))
                        sx = tf.cast(tf.schedule((tf.reshape(vs[:, 64:128], (8, 8, 16, 4)),), op=T.copy(rmem_layout=(8, 8 @ xu1.ti, 16 @ xu1.tj, 4))), "f32")
                        xv = sx * bq
                        xu1s = tf.schedule((tf.reshape(tf.cast(xv, "bf16"), (CH, 64)),), op=T.copy(smem_layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3)))
                    # phase 1, diagonal block 0: thread j of 16 holds column j of N = L + L N, rows in order
                    with Mesh(threads[0, 0:16], layout=(16,), names=("j",)) as p0:
                        xb0 = tf.zeros(Tensor[(16, 16), "f32", (16, 16 @ p0.j), "rmem"])
                        for i in range(16):
                            lk = tf.schedule((tf.reshape(asm[0 + i:0 + i + 1, 0:16], (16, 1)),), op=T.copy(rmem_layout=(16, 1)))
                            lij = tf.schedule((asm[0 + i:0 + i + 1, 0:16],), op=T.copy(rmem_layout=(1, 16 @ p0.j)))
                            sm = tf.schedule((tf.reduce(tf.schedule((lk * xb0,), op=T.copy(rmem_layout=((16, 16 @ p0.j), (1, 0)))), (0,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(1, 16 @ p0.j)))
                            xb0 = tf.insert_slice(xb0, lij + sm, (i, 0))
                        asm = tf.insert_slice(asm, xb0, (0, 0))
                    # phase 1, diagonal block 1: thread j of 16 holds column j of N = L + L N, rows in order
                    with Mesh(threads[0, 16:32], layout=(16,), names=("j",)) as p1:
                        xb1 = tf.zeros(Tensor[(16, 16), "f32", (16, 16 @ p1.j), "rmem"])
                        for i in range(16):
                            lk = tf.schedule((tf.reshape(asm[16 + i:16 + i + 1, 16:32], (16, 1)),), op=T.copy(rmem_layout=(16, 1)))
                            lij = tf.schedule((asm[16 + i:16 + i + 1, 16:32],), op=T.copy(rmem_layout=(1, 16 @ p1.j)))
                            sm = tf.schedule((tf.reduce(tf.schedule((lk * xb1,), op=T.copy(rmem_layout=((16, 16 @ p1.j), (1, 0)))), (0,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(1, 16 @ p1.j)))
                            xb1 = tf.insert_slice(xb1, lij + sm, (i, 0))
                        asm = tf.insert_slice(asm, xb1, (16, 16))
                    # phase 1, diagonal block 2: thread j of 16 holds column j of N = L + L N, rows in order
                    with Mesh(threads[1, 0:16], layout=(16,), names=("j",)) as p2:
                        xb2 = tf.zeros(Tensor[(16, 16), "f32", (16, 16 @ p2.j), "rmem"])
                        for i in range(16):
                            lk = tf.schedule((tf.reshape(asm[32 + i:32 + i + 1, 32:48], (16, 1)),), op=T.copy(rmem_layout=(16, 1)))
                            lij = tf.schedule((asm[32 + i:32 + i + 1, 32:48],), op=T.copy(rmem_layout=(1, 16 @ p2.j)))
                            sm = tf.schedule((tf.reduce(tf.schedule((lk * xb2,), op=T.copy(rmem_layout=((16, 16 @ p2.j), (1, 0)))), (0,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(1, 16 @ p2.j)))
                            xb2 = tf.insert_slice(xb2, lij + sm, (i, 0))
                        asm = tf.insert_slice(asm, xb2, (32, 32))
                    # phase 1, diagonal block 3: thread j of 16 holds column j of N = L + L N, rows in order
                    with Mesh(threads[1, 16:32], layout=(16,), names=("j",)) as p3:
                        xb3 = tf.zeros(Tensor[(16, 16), "f32", (16, 16 @ p3.j), "rmem"])
                        for i in range(16):
                            lk = tf.schedule((tf.reshape(asm[48 + i:48 + i + 1, 48:64], (16, 1)),), op=T.copy(rmem_layout=(16, 1)))
                            lij = tf.schedule((asm[48 + i:48 + i + 1, 48:64],), op=T.copy(rmem_layout=(1, 16 @ p3.j)))
                            sm = tf.schedule((tf.reduce(tf.schedule((lk * xb3,), op=T.copy(rmem_layout=((16, 16 @ p3.j), (1, 0)))), (0,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(1, 16 @ p3.j)))
                            xb3 = tf.insert_slice(xb3, lij + sm, (i, 0))
                        asm = tf.insert_slice(asm, xb3, (48, 48))
                    # phase 2a, super-block 0, computed transposed (j, i) as stored above the diagonal:
                    # X^T = L21^T + N11^T L21^T, then T21^T = X^T + X^T N22^T. Thread (ti, tj): column tj, rows ti, ti + 8.
                    with Mesh(threads, layout=(8, 16), names=("ti", "tj")) as e0:
                        n11t = tf.schedule((tf.reshape(tf.bitcast(asm[0:16, 0:16], layout=Layout((16, 16), (1, 65))), (16, 1, 16)),), op=T.copy(rmem_layout=(16 @ e0.tj, 1, 16)))
                        l21 = tf.schedule((tf.reshape(asm[16:32, 0:16], (1, 16, 16)),), op=T.copy(rmem_layout=(1, 2, 8 @ e0.ti, 16)))
                        l21t = tf.schedule((tf.reshape(tf.bitcast(asm[16:32, 0:16], layout=Layout((16, 16), (1, 65)) + 1040), (16, 16, 1)),), op=T.copy(rmem_layout=(16 @ e0.tj, 2, 8 @ e0.ti, 1)))
                        xt0 = l21t + tf.schedule((tf.reduce(tf.schedule((n11t * l21,), op=T.copy(rmem_layout=((16 @ e0.tj, 2, 8 @ e0.ti, 16), (0, 16, 0, 1)))), (2,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(16 @ e0.tj, 2, 8 @ e0.ti, 1)))
                        asm = tf.insert_slice(asm, tf.reshape(xt0, (16, 16)), (0, 16))
                        xs = tf.schedule((tf.reshape(asm[0:16, 16:32], (16, 1, 16)),), op=T.copy(rmem_layout=(16 @ e0.tj, 1, 16)))
                        n22 = tf.schedule((tf.reshape(asm[16:32, 16:32], (1, 16, 16)),), op=T.copy(rmem_layout=(1, 2, 8 @ e0.ti, 16)))
                        t21t = xt0 + tf.schedule((tf.reduce(tf.schedule((xs * n22,), op=T.copy(rmem_layout=((16 @ e0.tj, 2, 8 @ e0.ti, 16), (0, 16, 0, 1)))), (2,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(16 @ e0.tj, 2, 8 @ e0.ti, 1)))
                        asm = tf.insert_slice(asm, tf.reshape(t21t, (16, 16)), (0, 16))
                    # phase 2a, super-block 1, computed transposed (j, i) as stored above the diagonal:
                    # X^T = L21^T + N11^T L21^T, then T21^T = X^T + X^T N22^T. Thread (ti, tj): column tj, rows ti, ti + 8.
                    with Mesh(threads, layout=(8, 16), names=("ti", "tj")) as e1:
                        n11t = tf.schedule((tf.reshape(tf.bitcast(asm[32:48, 32:48], layout=Layout((16, 16), (1, 65)) + 2112), (16, 1, 16)),), op=T.copy(rmem_layout=(16 @ e1.tj, 1, 16)))
                        l21 = tf.schedule((tf.reshape(asm[48:64, 32:48], (1, 16, 16)),), op=T.copy(rmem_layout=(1, 2, 8 @ e1.ti, 16)))
                        l21t = tf.schedule((tf.reshape(tf.bitcast(asm[48:64, 32:48], layout=Layout((16, 16), (1, 65)) + 3152), (16, 16, 1)),), op=T.copy(rmem_layout=(16 @ e1.tj, 2, 8 @ e1.ti, 1)))
                        xt1 = l21t + tf.schedule((tf.reduce(tf.schedule((n11t * l21,), op=T.copy(rmem_layout=((16 @ e1.tj, 2, 8 @ e1.ti, 16), (0, 16, 0, 1)))), (2,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(16 @ e1.tj, 2, 8 @ e1.ti, 1)))
                        asm = tf.insert_slice(asm, tf.reshape(xt1, (16, 16)), (32, 48))
                        xs = tf.schedule((tf.reshape(asm[32:48, 48:64], (16, 1, 16)),), op=T.copy(rmem_layout=(16 @ e1.tj, 1, 16)))
                        n22 = tf.schedule((tf.reshape(asm[48:64, 48:64], (1, 16, 16)),), op=T.copy(rmem_layout=(1, 2, 8 @ e1.ti, 16)))
                        t21t = xt1 + tf.schedule((tf.reduce(tf.schedule((xs * n22,), op=T.copy(rmem_layout=((16 @ e1.tj, 2, 8 @ e1.ti, 16), (0, 16, 0, 1)))), (2,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(16 @ e1.tj, 2, 8 @ e1.ti, 1)))
                        asm = tf.insert_slice(asm, tf.reshape(t21t, (16, 16)), (32, 48))
                    # phase 2b, the lower-left 32 x 32 block: Y = L_BA (I + M_AA), T_BA = (I + M_BB) Y, with M the
                    # strictly lower part of each 32 x 32 diagonal block of T (N inside 16-blocks, T21 across them,
                    # read transposed); T_BA computed transposed, as stored. f32 products summed per thread;
                    # thread (ti, tj): rows ti + 8 e, columns tj + 16 f.
                    maa = tf.zeros(Tensor[(32, 33), "f32", Layout((32, 33), (33, 1)), "smem"])
                    maa = tf.insert_slice(maa, asm[0:16, 0:16], (0, 0))
                    maa = tf.insert_slice(maa, asm[16:32, 16:32], (16, 16))
                    maa = tf.insert_slice(maa, tf.bitcast(asm[0:16, 16:32], layout=Layout((16, 16), (1, 65)) + 16), (16, 0))
                    mbb = tf.zeros(Tensor[(32, 33), "f32", Layout((32, 33), (33, 1)), "smem"])
                    mbb = tf.insert_slice(mbb, asm[32:48, 32:48], (0, 0))
                    mbb = tf.insert_slice(mbb, asm[48:64, 48:64], (16, 16))
                    mbb = tf.insert_slice(mbb, tf.bitcast(asm[32:48, 48:64], layout=Layout((16, 16), (1, 65)) + 2128), (16, 0))
                    with Mesh(threads, layout=(8, 16), names=("ti", "tj")) as f:
                        lba = tf.schedule((tf.reshape(asm[32:64, 0:32], (32, 1, 32)),), op=T.copy(rmem_layout=(4, 8 @ f.ti, 1, 32)))
                        lbaij = tf.schedule((tf.reshape(asm[32:64, 0:32], (32, 32, 1)),), op=T.copy(rmem_layout=(4, 8 @ f.ti, 2, 16 @ f.tj, 1)))
                        maat = tf.schedule((tf.reshape(tf.bitcast(maa[:, 0:32], layout=Layout((32, 32), (1, 33))), (1, 32, 32)),), op=T.copy(rmem_layout=(1, 2, 16 @ f.tj, 32)))
                        y = lbaij + tf.schedule((tf.reduce(tf.schedule((lba * maat,), op=T.copy(rmem_layout=((4, 8 @ f.ti, 2, 16 @ f.tj, 32), (64, 0, 32, 0, 1)))), (2,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(4, 8 @ f.ti, 2, 16 @ f.tj, 1)))
                        ys = tf.insert_slice(tf.zeros(Tensor[(32, 33), "f32", Layout((32, 33), (33, 1)), "smem"]), tf.reshape(y, (32, 32)), (0, 0))
                        ytj = tf.schedule((tf.reshape(tf.bitcast(ys[:, 0:32], layout=Layout((32, 32), (1, 33))), (32, 1, 32)),), op=T.copy(rmem_layout=(2, 16 @ f.tj, 1, 32)))
                        mbi = tf.schedule((tf.reshape(mbb[:, 0:32], (1, 32, 32)),), op=T.copy(rmem_layout=(1, 4, 8 @ f.ti, 32)))
                        ytji = tf.schedule((tf.reshape(tf.bitcast(ys[:, 0:32], layout=Layout((32, 32), (1, 33))), (32, 32, 1)),), op=T.copy(rmem_layout=(2, 16 @ f.tj, 4, 8 @ f.ti, 1)))
                        tbat = ytji + tf.schedule((tf.reduce(tf.schedule((ytj * mbi,), op=T.copy(rmem_layout=((2, 16 @ f.tj, 4, 8 @ f.ti, 32), (128, 0, 32, 0, 1)))), (2,), True, ReduceKind.SUM),), op=T.copy(rmem_layout=(2, 16 @ f.tj, 4, 8 @ f.ti, 1)))
                        asm = tf.insert_slice(asm, tf.reshape(tbat, (32, 32)), (0, 32))
                    # T = I + N, read back as bf16: N inside 16-blocks below the diagonal, transposed storage across them
                    with Mesh(threads, layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as s2:
                        ir2 = tf.schedule((tf.reshape(iota, (CH, 1)),), op=T.copy(rmem_layout=((8 @ s2.lane8, 2, 4 @ s2.warp, 1), (1, 8, 16, 0))))
                        jc2 = tf.schedule((tf.reshape(iota, (1, CH)),), op=T.copy(rmem_layout=((1, 2, 4 @ s2.lane4, 8), (0, 64, 128, 512))))
                        zi2 = tf.zeros(Tensor[(1, 1), "i32", (1, 1), "rmem"])
                        c16 = tf.cast(tf.zeros(Tensor[(1, 1), "f32", (1, 1), "rmem"]) + 16.0, "i32")   # i32 sixteen
                        d2 = tf.schedule((ir2 - jc2,), op=T.copy(rmem_layout=((4 @ s2.warp, 2, 8 @ s2.lane8, 8, 4 @ s2.lane4, 2,), (16, 8, 1, 512, 128, 64))))
                        db = tf.schedule((ir2 // c16 - jc2 // c16,), op=T.copy(rmem_layout=((4 @ s2.warp, 2, 8 @ s2.lane8, 8, 4 @ s2.lane4, 2,), (16, 8, 1, 512, 128, 64))))
                        an = tf.schedule((asm[:, 0:CH],), op=T.copy(rmem_layout=((4 @ s2.warp, 2, 8 @ s2.lane8, 8, 4 @ s2.lane4, 2,), (16, 8, 1, 512, 128, 64))))
                        at = tf.schedule((tf.bitcast(asm[:, 0:CH], layout=Layout((64, 64), (1, 65))),), op=T.copy(rmem_layout=((4 @ s2.warp, 2, 8 @ s2.lane8, 8, 4 @ s2.lane4, 2,), (16, 8, 1, 512, 128, 64))))
                        tm = tf.where(d2 == zi2, 1.0, tf.where(d2 > zi2, tf.where(db == zi2, an, at), 0.0))
                        tb = tf.schedule((tf.cast(tm, "bf16"),), op=T.copy(smem_layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3)))
                    # w = T bf16(beta k e^g), u = T bf16(beta v), one 64-wide column chunk at a time on the tensor
                    # cores (A = T K-major, B = the X chunk MN-major); each 64 x 128 result staged in SW128 smem and
                    # stored by TMA into the packed row.
                    ost = tf.zeros(Tensor[(CH, D), "bf16", Layout((CH, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3), "smem"])
                    with Mesh(threads, layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as mw0:
                        ow0 = tf.schedule((tf.zeros(Tensor[(CH, 64), "f32", ((4 @ mw0.warp, 2, 8 @ mw0.lane8, 8, 4 @ mw0.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"]), tb, xw0s),
                                             op=T.tiled_mma(atom=w_tx), repeat=(1, 1, 4))
                        ost = tf.insert_slice(ost, tf.cast(ow0, "bf16"), (0, 0))
                    with Mesh(threads, layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as mw1:
                        ow1 = tf.schedule((tf.zeros(Tensor[(CH, 64), "f32", ((4 @ mw1.warp, 2, 8 @ mw1.lane8, 8, 4 @ mw1.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"]), tb, xw1s),
                                             op=T.tiled_mma(atom=w_tx), repeat=(1, 1, 4))
                        ost = tf.insert_slice(ost, tf.cast(ow1, "bf16"), (0, 64))
                    with threads[0, :] as _storer:
                        tile = tf.schedule((ost,), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (t0, cta.h * WUP))
                    ost = tf.zeros(Tensor[(CH, D), "bf16", Layout((CH, (2, 64)), (64, (4096, 1))) | Swizzle(3, 4, 3), "smem"])
                    with Mesh(threads, layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as mu0:
                        ou0 = tf.schedule((tf.zeros(Tensor[(CH, 64), "f32", ((4 @ mu0.warp, 2, 8 @ mu0.lane8, 8, 4 @ mu0.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"]), tb, xu0s),
                                             op=T.tiled_mma(atom=w_tx), repeat=(1, 1, 4))
                        ost = tf.insert_slice(ost, tf.cast(ou0, "bf16"), (0, 0))
                    with Mesh(threads, layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as mu1:
                        ou1 = tf.schedule((tf.zeros(Tensor[(CH, 64), "f32", ((4 @ mu1.warp, 2, 8 @ mu1.lane8, 8, 4 @ mu1.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"]), tb, xu1s),
                                             op=T.tiled_mma(atom=w_tx), repeat=(1, 1, 4))
                        ost = tf.insert_slice(ost, tf.cast(ou1, "bf16"), (0, 64))
                    with threads[0, :] as _storer:
                        tile = tf.schedule((ost,), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (t0, cta.h * WUP + D))
                    return out

        @func
        def entry(q: Tensor[(1, M, HK, D), "bf16"], k: Tensor[(1, M, HK, D), "bf16"], v: Tensor[(1, M, HV, D), "bf16"],
                  g: Tensor[(1, M, HV), "f32"], beta: Tensor[(1, M, HV), "bf16"]):
            iota = tf.arange(type=Tensor[(CH,), "i32"], start=0, step=1)
            q3 = tf.reshape(q, (M, HK, D))
            k3 = tf.reshape(k, (M, HK, D))
            gc = tf.reshape(intra_g(tf.reshape(g, (M, HV)), iota), (HV, M))  # noqa: F821
            wup = intra_wup(q3, k3, tf.reshape(v, (M, HV, D)), gc, tf.reshape(beta, (M, HV)), iota)  # noqa: F821
            return gc, tf.reshape(wup, (M, HV, WUP))

    DeltaRuleIntra.__doc__ = f"delta_rule, intra part, at seq_len={M}: ({C}, {HV}) CTAs x 128 threads"
    return DeltaRuleIntra


def delta_rule_recur(M: int):
    """Invocation 2 of model.py's delta_rule: o, every chunk of a head in order
    (it reads all chunks' w / u / P, so it follows invocation 1 -- the
    launch boundary is that grid-wide dependence)."""
    C = M // CH

    @module(entry="recur_out", target=_H200, topologies=(Topology("cta", NV * HV), Topology("thread", 384)))
    class DeltaRuleRecur:
        @func
        def recur_out(q: Tensor[(M, HK, D), "bf16"], k: Tensor[(M, HK, D), "bf16"], wup: Tensor[(M, HV, WUP), "bf16"],
                      gc: Tensor[(HV, M), "f32"]):
            with Mesh(("cta",), layout=(NV, HV), names=("vb", "h")) as cta:
                with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                    w_kmn = T.cuda.sm90.Wgmma(n=BV, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)
                    w_mnmn = T.cuda.sm90.Wgmma(n=BV, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.MN)
                    out = tf.zeros(Tensor[(M, HV, D), "bf16"])
                    kh = cta.h // VPK
                    v0 = cta.vb * BV
                    # WG0: the state S (d x v, f32), and its bf16 copy for the tensor cores
                    with Mesh(threads[0, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as s0:
                        st = tf.zeros(Tensor[(D, BV), "f32", ((2, 4 @ s0.warp, 2, 8 @ s0.lane8, 8, 4 @ s0.lane4, 2,), (4096, 16, 8, 1, 512, 128, 64)), "rmem"])
                    for c in range(C):
                        t0 = c * CH
                        # WG0: S_c in bf16 for the tensor cores (3-deep: WG1 reads it a chunk behind)
                        with Mesh(threads[0, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as s0:
                            sb = tf.schedule((tf.cast(st, "bf16"),), op=T.copy(smem_layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3)), buffers=3)
                        # WG2: the loads of chunk c, double-buffered
                        with threads[2, :32] as _loader:
                            ws = tf.schedule((tf.reshape(wup[t0:t0 + CH, cta.h:cta.h + 1, 0:D], (CH, D)),), op=T.copy_async_tensor(smem_layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3)), buffers=2)
                            ks = tf.schedule((tf.reshape(k[t0:t0 + CH, kh:kh + 1, :], (CH, D)),), op=T.copy_async_tensor(smem_layout=Layout(((4, 2, 8), (2, 64)), ((1024, 512, 64), (4096, 1))) | Swizzle(3, 4, 3)), buffers=2)
                            qs = tf.schedule((tf.reshape(q[t0:t0 + CH, kh:kh + 1, :], (CH, D)),), op=T.copy_async_tensor(smem_layout=Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3)), buffers=2)
                            us = tf.schedule((tf.reshape(wup[t0:t0 + CH, cta.h:cta.h + 1, D + v0:D + v0 + BV], (CH, BV)),), op=T.copy_async_tensor(smem_layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3)), buffers=2)
                            ps = tf.schedule((tf.reshape(wup[t0:t0 + CH, cta.h:cta.h + 1, 2 * D:2 * D + CH], (CH, CH)),), op=T.copy_async_tensor(smem_layout=Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=2)
                            gs = tf.schedule((gc[cta.h:cta.h + 1, t0:t0 + CH],), op=T.copy_async_tensor(smem_layout=Layout((1, CH), (CH, 1))), buffers=2)
                        # WG0, 64 threads: per-row decays of chunk c, shared with WG1
                        with Mesh(threads[0, 0:64], layout=(64,), names=("t",)) as d0:
                            g64 = tf.schedule((tf.reshape(gs, (CH,)),), op=T.copy(rmem_layout=(CH @ d0.t,)))
                            gl = tf.schedule((tf.reshape(gs[0:1, CH - 1:CH], (1,)),), op=T.copy(rmem_layout=(1,)))
                            eds = tf.schedule((tf.exp(gl - g64),), op=T.copy(smem_layout=Layout((CH,), (1,))), buffers=3)
                            eqs = tf.schedule((tf.exp(g64) * SCALE,), op=T.copy(smem_layout=Layout((CH,), (1,))), buffers=3)
                        # WG0: v_new = u - w S;  S <- e^gl S + K^T (v_new e^(gl - g))
                        with Mesh(threads[0, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as s0:
                            vn = tf.zeros(Tensor[(CH, BV), "f32", ((4 @ s0.warp, 2, 8 @ s0.lane8, 8, 4 @ s0.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"])
                            vn = tf.schedule((vn, ws, tf.bitcast(sb, layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=w_kmn), repeat=(1, 1, 8))
                            el = tf.exp(tf.schedule((tf.reshape(gs[0:1, CH - 1:CH], (1, 1)),), op=T.copy(rmem_layout=(1, 1))))
                            st = st * el
                            uu = tf.cast(tf.schedule((us,), op=T.copy(rmem_layout=((4 @ s0.warp, 2, 8 @ s0.lane8, 8, 4 @ s0.lane4, 2,), (16, 8, 1, 512, 128, 64)))), "f32")
                            vn = uu - vn
                            edr = tf.schedule((tf.reshape(eds, (CH, 1)),), op=T.copy(rmem_layout=((8 @ s0.lane8, 2, 4 @ s0.warp, 1), (1, 8, 16, 0))))
                            vnb = tf.schedule((tf.cast(vn, "bf16"),), op=T.copy(smem_layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3)), buffers=3)
                            vd = tf.schedule((tf.cast(vn * edr, "bf16"),), op=T.copy(smem_layout=Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3)))
                            st = tf.schedule((st, tf.bitcast(ks, layout=Layout(((2, 1, 64), (4, 2, 8)), ((4096, 1024, 1), (1024, 512, 64))) | Swizzle(3, 4, 3)), vd), op=T.tiled_mma(atom=w_mnmn), repeat=(2, 1, 4))
                        # WG1: o = scale e^g (q S_c) + P v_new, staged and stored by TMA
                        with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as s1:
                            o = tf.zeros(Tensor[(CH, BV), "f32", ((4 @ s1.warp, 2, 8 @ s1.lane8, 8, 4 @ s1.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"])
                            o = tf.schedule((o, qs, tf.bitcast(sb, layout=Layout(((8, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=w_kmn), repeat=(1, 1, 8))
                            eqr = tf.schedule((tf.reshape(eqs, (CH, 1)),), op=T.copy(rmem_layout=((8 @ s1.lane8, 2, 4 @ s1.warp, 1), (1, 8, 16, 0))))
                            o = o * eqr
                            o = tf.schedule((o, ps, vnb), op=T.tiled_mma(atom=w_kmn), repeat=(1, 1, 4))
                            os = tf.schedule((tf.cast(o, "bf16"),), op=T.copy(smem_layout=Layout(((8, 8), 64), ((512, 64), 1)) | Swizzle(3, 4, 3)), buffers=2)
                        with threads[1, :32] as _storer:
                            tile = tf.schedule((tf.reshape(os, (CH, 1, BV)),), op=T.copy_async_tensor())
                            out = tf.insert_slice(out, tile, (t0, cta.h, v0))
                    return out

    DeltaRuleRecur.__doc__ = f"delta_rule, recurrence, at seq_len={M}: ({NV}, {HV}) CTAs x 384 threads"
    return DeltaRuleRecur


# model.py's delta_rule(q, k, v, g, beta) = host orchestration of two invocations:
#   gc, wup = DeltaRuleIntra{n}.entry(q, k, v, g, beta)
#   o = DeltaRuleRecur{n}.recur_out(q[0], k[0], wup, gc)
for _M in (1024, 4096, 8192):
    globals()[f"DeltaRuleIntra{_M}"] = delta_rule_intra(_M)
    globals()[f"DeltaRuleRecur{_M}"] = delta_rule_recur(_M)
