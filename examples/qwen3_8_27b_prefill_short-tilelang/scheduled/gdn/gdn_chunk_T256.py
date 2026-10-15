"""Scheduled HIR: GDN chunk (intra-chunk WY quantities) at prompt length 256 (NC = 4 chunks).

Implements model.py Qwen3_8DeltaNetLayer.delta_rule, the per-chunk part (everything
before the sequential pass over chunks), for one value head h and chunk c, with
G = in-chunk cumulative sum of g (kept in log2 units, G' = G * log2(e)):
  A   = strict_lower(beta_i * exp(G_i - G_j) * k_i.k_j)
  X   = (I + A)^-1        (the HIR's forward substitution of -A, plus I, is this X)
  U^T = V^T (X diag(beta))^T               (HIR `value`, transposed, f32)
  W   = X diag(beta e^G) K                  (HIR `k_cumdecay`, as bf16 hi + lo)
  P   = lower(q_i.k_j * exp(G_i - G_j)) / sqrt(128)   (HIR `scores` with q scaled, bf16)
  F   = [exp(G_63 - G_j), exp(G_j) / sqrt(128), exp(G_63)]   (the chunk decay factors)
Runtime kernel: runtime/kernels/gdn.py make_chunk(256) (`gdn_chunk`, precise state path).

Decomposition (as the kernel): CTA mesh (value head h, chunk c), c fastest (tilelang
grid (NC, 48)); key head kh = h // 3. 256 threads = 2 warpgroups.
- One warp issues the TMA loads of the K, Q, V tiles (64 x 128 bf16, two 64-column
  128B-swizzled panels each).
- Warp 0 computes G (lane l owns rows 2l, 2l+1), threads 32..95 load beta.
- KK = K K^T and QK = Q K^T: wgmma SS m64n32k16, K-major, warpgroup g owns columns
  [32g, 32g+32) (the kernel's tilelang partition: N split over the warpgroups).
- In the accumulator registers: dec = exp2(min(G'_i - G'_j, 0)),
  A = beta_i * dec * KK (strictly lower), P = bf16(QK * dec / sqrt(128)) (lower).
  A goes to a padded (64, 68) f32 shared tile, P is staged in shared memory and
  TMA-stored. Threads 0..63 compute exp2(G'), BG = beta * exp2(G') and F (TMA-stored).
- X = (I + A)^-1 in f32 SIMT on shared memory: the four 16x16 diagonal blocks by
  forward substitution (threads 0..63: 16 per block, thread = one column, 16 rows in
  registers), then two doubling levels (X21 = -X22 A21 X11), each as a Y phase into a
  padded Y tile and an X phase:
    level 16 -> 32, blocks (1,0) on warpgroup 0 and (3,2) on warpgroup 1, thread =
      (row, column pair), 1x2 register tile, 16-term sums;
    level 32 -> 64, Y = A[32:64, 0:32] X[0:32, 0:32] with warpgroup g owning columns
      [16g, 16g+16), then X[32:64, 0:32] = -X[32:64, 32:64] Y with warpgroup 0 owning
      rows 48..63 and warpgroup 1 rows 32..47; 2x2 register tiles, 32-term sums (the
      cleared upper blocks of X add exact zeros).
- Mb = X diag(beta), Mg = X diag(beta e^G), each split into bf16 hi + lo, stored
  as 128B-swizzled (64, 64) tiles (256 threads, 4 rows x float4 each).
- U^T = V^T Mbh^T + V^T Mbl^T: wgmma SS m64n32k16 with V^T MN-major from the V tile
  (2 M atoms per warpgroup, warpgroup g owns columns [32g, 32g+32)); W = Mgh K + Mgl K:
  wgmma SS m64n64k16 with K MN-major from the same K tile (warpgroup g owns dk columns
  [64g, 64g+64)). W is split into bf16 hi and lo; U^T, W hi, W lo are staged in
  swizzled shared tiles and TMA-stored.

Differences to the kernel:
- Result: scheduled lowering accepts one tensor result. The kernel's four workspaces
  (U^T (48, NC, 128, 64) f32, W (48, NC, 2, 64, 128) bf16, P (48, NC, 64, 64) bf16,
  F (48, NC, 3, 64) f32) are one f32 tensor (48, NC, 28864): per (h, c), [0, 8192) =
  U^T (128 x 64 row-major), [8192, 16384) = W hi (64 x 128), [16384, 24576) = W lo,
  [24576, 28672) = P (64 x 64), [28672, 28864) = F (3 x 64). W hi/lo and P keep their
  bf16 rounding and are widened to f32 exactly before staging, so their staging tiles
  are f32 (32-column panels) where the kernel stages and TMA-stores bf16 tiles.
- G: the kernel scans g with shfl_up (Hillis-Steele over 32 lanes, 2 rows per lane).
  TileFoundry has no shuffle; this program states the inclusive cumsum as the masked
  row sum G_i = sum_j g_j [j <= i] (the HIR's product with the triangular ones), lane l
  computing its rows 2l, 2l+1 from all 64 g values. Same values up to f32 summation
  order.
- Masks: the kernel evaluates i > j, i >= j and i == j from indices
  (T.if_then_else); this program reads them from a constant input `tri` (3, 64, 64)
  f32 ([0] = (j <= i), [1] = (j < i), [2] = identity) and multiplies by them
  (tf.arange has no TIR lowering). Products with 0/1 give the same values (+0 vs -0
  only).
- Broadcast operands (G'_i, G'_j, beta_i in the accumulator layout; exp2(G'_63)) are
  register copies read from the shared G / beta vectors; the kernel reads G_s[i],
  G_s[j], B_s[i] from shared memory inside its element loop. The row and column
  vectors are added to a zero accumulator first so every elementwise operand carries
  the accumulator layout (+ 0.0 changes no value).
- Forward substitution: the kernel subtracts A[i, j] * x[j] term by term from (i == j);
  this program subtracts the row sum; same terms, f32 order differs. The kernel unrolls
  the 16 steps; here they are one loop.
- Inverse level 16 -> 32: the kernel runs the Y phase of both blocks on all 256 threads,
  synchronises, then the X phase; here the Y and X steps of one block are one region on
  its warpgroup (same thread -> element mapping, same values).
- In-place reuse: the kernel's shared allocator aliases buffers whose lifetimes do not
  overlap (e.g. UT_s on K_s, W_s on V_s); this program names every buffer once.
- Synchronisation (mbarrier waits on the TMA loads, __syncthreads between phases,
  wgmma fences/waits) is not stated; TileFoundry derives it from the data flow.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

T_ = 256
NC = T_ // 64
HV = 48
D = 128
C = 64
LOG2E = 1.4426950408889634
SCALE = 0.08838834764831845
U_OFF = 0
WH_OFF = 8192
WL_OFF = 16384
P_OFF = 24576
F_OFF = 28672
PACK = 28864


@module(entry="gdn_chunk", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", HV * NC), Topology("thread", 256)))
class SCHED_GDN_CHUNK_T256:
    @func
    def gdn_chunk(q: Tensor[(T_, 16, D), "bf16"], k: Tensor[(T_, 16, D), "bf16"], v: Tensor[(T_, HV, D), "bf16"],
                  g: Tensor[(T_, HV), "f32"], beta: Tensor[(T_, HV), "bf16"],
                  tri: Tensor[(3, C, C), "f32"]) -> Tensor[(HV, NC, PACK), "f32"]:
        with Mesh(("cta",), layout=(HV, NC), names=("h", "c")) as cta:
            with Mesh(("thread",), layout=(2, 128), names=("warpgroup", "participant")) as threads:
                s_mma = T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                u_mma = T.cuda.sm90.Wgmma(n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.MN, b_major=T.cuda.sm90.Major.K)
                w_mma = T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)
                out = tf.zeros(Tensor[(HV, NC, PACK), "f32"])
                kh = cta.h // 3
                t0 = cta.c * C
                with threads[0, :32] as _ld:
                    ks = tf.schedule((tf.reshape(k[t0:t0 + C, kh:kh + 1, :], new_shape=(C, D)), tf.empty(Tensor[(C, D), "bf16", Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy_async_tensor())
                    qs = tf.schedule((tf.reshape(q[t0:t0 + C, kh:kh + 1, :], new_shape=(C, D)), tf.empty(Tensor[(C, D), "bf16", Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy_async_tensor())
                    vs = tf.schedule((tf.reshape(v[t0:t0 + C, cta.h:cta.h + 1, :], new_shape=(C, D)), tf.empty(Tensor[(C, D), "bf16", Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy_async_tensor())
                Xs = tf.zeros(Tensor[(C, C), "f32", Layout((C, C), (68, 1)), "smem"])
                Ys = tf.empty(Tensor[(C, C), "f32", Layout((C, C), (68, 1)), "smem"])
                with Mesh(threads[0, 0:32], layout=(32,), names=("l",)) as wl:
                    gr = tf.schedule((tf.reshape(g[t0:t0 + C, cta.h:cta.h + 1], new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", (1, C), "rmem"])), op=T.copy())
                    lm = tf.schedule((tf.reshape(tri[0:1, :, :], new_shape=(C, C)), tf.empty(Tensor[(C, C), "f32", ((32 @ wl.l, 2, C), (0, C, 1)), "rmem"])), op=T.copy())
                    G = tf.reduce(lm * gr, axes=(-1,), keepdim=True, kind="sum") * LOG2E
                    Gs = tf.schedule((G, tf.empty(Tensor[(C, 1), "f32", Layout((C, 1), (1, 1)), "smem"])), op=T.copy())
                with Mesh(threads[0, 32:96], layout=(64,), names=("r",)) as wb:
                    bb = tf.schedule((beta[t0:t0 + C, cta.h:cta.h + 1], tf.empty(Tensor[(C, 1), "bf16", ((64 @ wb.r, 1), (0, 0)), "rmem"])), op=T.copy())
                    Bs = tf.schedule((tf.cast(bb, dtype="f32"), tf.empty(Tensor[(C, 1), "f32", Layout((C, 1), (1, 1)), "smem"])), op=T.copy())
                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as cw:
                    KK = tf.schedule((tf.zeros(Tensor[(C, C), "f32", ((2 @ cw.group, 8 @ cw.lane8, 2, 4 @ cw.warp, 2, 4 @ cw.lane4, 4), (2048, 1, 8, 16, 64, 128, 512)), "rmem"]), ks, tf.bitcast(ks, Layout(((2, 4, 16), (2, 4, 8)), ((4096, 16, 1), (2048, 512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=s_mma), repeat=(1, 2, 8))
                    QK = tf.schedule((tf.zeros(Tensor[(C, C), "f32", ((2 @ cw.group, 8 @ cw.lane8, 2, 4 @ cw.warp, 2, 4 @ cw.lane4, 4), (2048, 1, 8, 16, 64, 128, 512)), "rmem"]), qs, tf.bitcast(ks, Layout(((2, 4, 16), (2, 4, 8)), ((4096, 16, 1), (2048, 512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=s_mma), repeat=(1, 2, 8))
                with Mesh(threads[0, 0:64], layout=(64,), names=("r",)) as m64:
                    gi = tf.schedule((tf.reshape(Gs, new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", ((1, 64 @ m64.r), (0, 0)), "rmem"])), op=T.copy())
                    bi = tf.schedule((tf.reshape(Bs, new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", ((1, 64 @ m64.r), (0, 0)), "rmem"])), op=T.copy())
                    g63 = tf.schedule((Gs[C - 1:C, :], tf.empty(Tensor[(1, 1), "f32", (1, 1), "rmem"])), op=T.copy())
                    eg = tf.exp2(gi)
                    BGs = tf.schedule((bi * eg, tf.empty(Tensor[(1, C), "f32", Layout((1, C), (C, 1)), "smem"])), op=T.copy())
                    Fs = tf.empty(Tensor[(3, C), "f32", Layout((3, C), (C, 1)), "smem"])
                    Fs = tf.insert_slice(Fs, tf.exp2(tf.neg(gi) + g63), (0, 0))
                    Fs = tf.insert_slice(Fs, eg * SCALE, (1, 0))
                    Fs = tf.insert_slice(Fs, tf.zeros(Tensor[(1, C), "f32", ((1, 64 @ m64.r), (0, 0)), "rmem"]) + tf.exp2(g63), (2, 0))
                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as cw:
                    Gi = tf.schedule((Gs, tf.empty(Tensor[(C, 1), "f32", ((1, 8 @ cw.lane8, 2, 4 @ cw.warp, 1, 1, 1), (0, 1, 8, 16, 0, 0, 0)), "rmem"])), op=T.copy())
                    Gj = tf.schedule((tf.reshape(Gs, new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", ((2 @ cw.group, 1, 1, 1, 2, 4 @ cw.lane4, 4), (2048, 0, 0, 0, 64, 128, 512)), "rmem"])), op=T.copy())
                    Bi = tf.schedule((Bs, tf.empty(Tensor[(C, 1), "f32", ((1, 8 @ cw.lane8, 2, 4 @ cw.warp, 1, 1, 1), (0, 1, 8, 16, 0, 0, 0)), "rmem"])), op=T.copy())
                    lower_f = tf.schedule((tf.reshape(tri[0:1, :, :], new_shape=(C, C)), tf.empty(Tensor[(C, C), "f32", ((2 @ cw.group, 8 @ cw.lane8, 2, 4 @ cw.warp, 2, 4 @ cw.lane4, 4), (2048, 1, 8, 16, 64, 128, 512)), "rmem"])), op=T.copy())
                    strict_f = tf.schedule((tf.reshape(tri[1:2, :, :], new_shape=(C, C)), tf.empty(Tensor[(C, C), "f32", ((2 @ cw.group, 8 @ cw.lane8, 2, 4 @ cw.warp, 2, 4 @ cw.lane4, 4), (2048, 1, 8, 16, 64, 128, 512)), "rmem"])), op=T.copy())
                    dec = tf.exp2(tf.min(tf.zeros(Tensor[(C, C), "f32", ((2 @ cw.group, 8 @ cw.lane8, 2, 4 @ cw.warp, 2, 4 @ cw.lane4, 4), (2048, 1, 8, 16, 64, 128, 512)), "rmem"]) + Gi - Gj, 0.0))
                    Afr = (tf.zeros(Tensor[(C, C), "f32", ((2 @ cw.group, 8 @ cw.lane8, 2, 4 @ cw.warp, 2, 4 @ cw.lane4, 4), (2048, 1, 8, 16, 64, 128, 512)), "rmem"]) + Bi) * dec * KK * strict_f
                    Pfr = tf.cast(tf.cast(QK * dec * SCALE * lower_f, dtype="bf16"), dtype="f32")
                    As = tf.schedule((Afr, tf.empty(Tensor[(C, C), "f32", Layout((C, C), (68, 1)), "smem"])), op=T.copy())
                    Pst = tf.schedule((Pfr, tf.empty(Tensor[(C, C), "f32", Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy())
                with threads[0, :32] as _st:
                    po = tf.schedule((Pst, tf.empty(Tensor[(C, C), "f32", (C, C)])), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tf.reshape(po, new_shape=(1, 1, C * C)), (cta.h, cta.c, P_OFF))
                    fo = tf.schedule((Fs, tf.empty(Tensor[(3, C), "f32", (3, C)])), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tf.reshape(fo, new_shape=(1, 1, 3 * C)), (cta.h, cta.c, F_OFF))
                with Mesh(threads[0, 0:16], layout=(16,), names=("cc",)) as md0:
                    Xb = tf.zeros(Tensor[(16, 16), "f32", ((16, 16 @ md0.cc), (1, 0)), "rmem"])
                    for i in range(16):
                        ai = tf.schedule((tf.reshape(As[0 + i:0 + i + 1, 0:16], new_shape=(16, 1)), tf.empty(Tensor[(16, 1), "f32", (16, 1), "rmem"])), op=T.copy())
                        ei = tf.schedule((tf.reshape(tri[2:3, i:i + 1, 0:16], new_shape=(1, 16)), tf.empty(Tensor[(1, 16), "f32", ((1, 16 @ md0.cc), (0, 0)), "rmem"])), op=T.copy())
                        Xb = tf.insert_slice(Xb, ei - tf.reduce(Xb * ai, axes=(0,), keepdim=True, kind="sum"), (i, 0))
                    Xs = tf.insert_slice(Xs, Xb, (0, 0))
                with Mesh(threads[0, 16:32], layout=(16,), names=("cc",)) as md1:
                    Xb = tf.zeros(Tensor[(16, 16), "f32", ((16, 16 @ md1.cc), (1, 0)), "rmem"])
                    for i in range(16):
                        ai = tf.schedule((tf.reshape(As[16 + i:16 + i + 1, 16:32], new_shape=(16, 1)), tf.empty(Tensor[(16, 1), "f32", (16, 1), "rmem"])), op=T.copy())
                        ei = tf.schedule((tf.reshape(tri[2:3, i:i + 1, 0:16], new_shape=(1, 16)), tf.empty(Tensor[(1, 16), "f32", ((1, 16 @ md1.cc), (0, 0)), "rmem"])), op=T.copy())
                        Xb = tf.insert_slice(Xb, ei - tf.reduce(Xb * ai, axes=(0,), keepdim=True, kind="sum"), (i, 0))
                    Xs = tf.insert_slice(Xs, Xb, (16, 16))
                with Mesh(threads[0, 32:48], layout=(16,), names=("cc",)) as md2:
                    Xb = tf.zeros(Tensor[(16, 16), "f32", ((16, 16 @ md2.cc), (1, 0)), "rmem"])
                    for i in range(16):
                        ai = tf.schedule((tf.reshape(As[32 + i:32 + i + 1, 32:48], new_shape=(16, 1)), tf.empty(Tensor[(16, 1), "f32", (16, 1), "rmem"])), op=T.copy())
                        ei = tf.schedule((tf.reshape(tri[2:3, i:i + 1, 0:16], new_shape=(1, 16)), tf.empty(Tensor[(1, 16), "f32", ((1, 16 @ md2.cc), (0, 0)), "rmem"])), op=T.copy())
                        Xb = tf.insert_slice(Xb, ei - tf.reduce(Xb * ai, axes=(0,), keepdim=True, kind="sum"), (i, 0))
                    Xs = tf.insert_slice(Xs, Xb, (32, 32))
                with Mesh(threads[0, 48:64], layout=(16,), names=("cc",)) as md3:
                    Xb = tf.zeros(Tensor[(16, 16), "f32", ((16, 16 @ md3.cc), (1, 0)), "rmem"])
                    for i in range(16):
                        ai = tf.schedule((tf.reshape(As[48 + i:48 + i + 1, 48:64], new_shape=(16, 1)), tf.empty(Tensor[(16, 1), "f32", (16, 1), "rmem"])), op=T.copy())
                        ei = tf.schedule((tf.reshape(tri[2:3, i:i + 1, 0:16], new_shape=(1, 16)), tf.empty(Tensor[(1, 16), "f32", ((1, 16 @ md3.cc), (0, 0)), "rmem"])), op=T.copy())
                        Xb = tf.insert_slice(Xb, ei - tf.reduce(Xb * ai, axes=(0,), keepdim=True, kind="sum"), (i, 0))
                    Xs = tf.insert_slice(Xs, Xb, (48, 48))
                with Mesh(threads[0, :], layout=(16, 8), names=("r", "cp")) as ml0:
                    ar = tf.schedule((tf.reshape(As[16:32, 0:16], new_shape=(16, 16, 1)), tf.empty(Tensor[(16, 16, 1), "f32", ((16 @ ml0.r, 16, 1), (0, 1, 0)), "rmem"])), op=T.copy())
                    xc = tf.schedule((tf.reshape(Xs[0:16, 0:16], new_shape=(1, 16, 16)), tf.empty(Tensor[(1, 16, 16), "f32", ((1, 16, 8 @ ml0.cp, 2), (0, 2, 0, 1)), "rmem"])), op=T.copy())
                    Ys = tf.insert_slice(Ys, tf.reduce(ar * xc, axes=(1,), keepdim=False, kind="sum"), (16, 0))
                    dr = tf.schedule((tf.reshape(Xs[16:32, 16:32], new_shape=(16, 16, 1)), tf.empty(Tensor[(16, 16, 1), "f32", ((16 @ ml0.r, 16, 1), (0, 1, 0)), "rmem"])), op=T.copy())
                    yc = tf.schedule((tf.reshape(Ys[16:32, 0:16], new_shape=(1, 16, 16)), tf.empty(Tensor[(1, 16, 16), "f32", ((1, 16, 8 @ ml0.cp, 2), (0, 2, 0, 1)), "rmem"])), op=T.copy())
                    Xs = tf.insert_slice(Xs, tf.neg(tf.reduce(dr * yc, axes=(1,), keepdim=False, kind="sum")), (16, 0))
                with Mesh(threads[1, :], layout=(16, 8), names=("r", "cp")) as ml1:
                    ar = tf.schedule((tf.reshape(As[48:64, 32:48], new_shape=(16, 16, 1)), tf.empty(Tensor[(16, 16, 1), "f32", ((16 @ ml1.r, 16, 1), (0, 1, 0)), "rmem"])), op=T.copy())
                    xc = tf.schedule((tf.reshape(Xs[32:48, 32:48], new_shape=(1, 16, 16)), tf.empty(Tensor[(1, 16, 16), "f32", ((1, 16, 8 @ ml1.cp, 2), (0, 2, 0, 1)), "rmem"])), op=T.copy())
                    Ys = tf.insert_slice(Ys, tf.reduce(ar * xc, axes=(1,), keepdim=False, kind="sum"), (48, 32))
                    dr = tf.schedule((tf.reshape(Xs[48:64, 48:64], new_shape=(16, 16, 1)), tf.empty(Tensor[(16, 16, 1), "f32", ((16 @ ml1.r, 16, 1), (0, 1, 0)), "rmem"])), op=T.copy())
                    yc = tf.schedule((tf.reshape(Ys[48:64, 32:48], new_shape=(1, 16, 16)), tf.empty(Tensor[(1, 16, 16), "f32", ((1, 16, 8 @ ml1.cp, 2), (0, 2, 0, 1)), "rmem"])), op=T.copy())
                    Xs = tf.insert_slice(Xs, tf.neg(tf.reduce(dr * yc, axes=(1,), keepdim=False, kind="sum")), (48, 32))
                with Mesh(threads[0:2, :], layout=(2, 16, 8), names=("hw", "rp", "c8")) as m2y:
                    ar = tf.schedule((tf.reshape(As[32:64, 0:32], new_shape=(32, 32, 1)), tf.empty(Tensor[(32, 32, 1), "f32", ((16 @ m2y.rp, 2, 32, 1), (0, 32, 1, 0)), "rmem"])), op=T.copy())
                    xc = tf.schedule((tf.reshape(Xs[0:32, 0:32], new_shape=(1, 32, 32)), tf.empty(Tensor[(1, 32, 32), "f32", ((1, 32, 2 @ m2y.hw, 8 @ m2y.c8, 2), (0, 2, 0, 0, 1)), "rmem"])), op=T.copy())
                    Ys = tf.insert_slice(Ys, tf.reduce(ar * xc, axes=(1,), keepdim=False, kind="sum"), (32, 0))
                with Mesh(threads[0, :], layout=(8, 16), names=("r8", "cr")) as m2x0:
                    dr = tf.schedule((tf.reshape(Xs[48:64, 32:64], new_shape=(16, 32, 1)), tf.empty(Tensor[(16, 32, 1), "f32", ((8 @ m2x0.r8, 2, 32, 1), (0, 32, 1, 0)), "rmem"])), op=T.copy())
                    yc = tf.schedule((tf.reshape(Ys[32:64, 0:32], new_shape=(1, 32, 32)), tf.empty(Tensor[(1, 32, 32), "f32", ((1, 32, 16 @ m2x0.cr, 2), (0, 2, 0, 1)), "rmem"])), op=T.copy())
                    Xs = tf.insert_slice(Xs, tf.neg(tf.reduce(dr * yc, axes=(1,), keepdim=False, kind="sum")), (48, 0))
                with Mesh(threads[1, :], layout=(8, 16), names=("r8", "cr")) as m2x1:
                    dr = tf.schedule((tf.reshape(Xs[32:48, 32:64], new_shape=(16, 32, 1)), tf.empty(Tensor[(16, 32, 1), "f32", ((8 @ m2x1.r8, 2, 32, 1), (0, 32, 1, 0)), "rmem"])), op=T.copy())
                    yc = tf.schedule((tf.reshape(Ys[32:64, 0:32], new_shape=(1, 32, 32)), tf.empty(Tensor[(1, 32, 32), "f32", ((1, 32, 16 @ m2x1.cr, 2), (0, 2, 0, 1)), "rmem"])), op=T.copy())
                    Xs = tf.insert_slice(Xs, tf.neg(tf.reduce(dr * yc, axes=(1,), keepdim=False, kind="sum")), (32, 0))
                with Mesh(("thread",), layout=(16, 16), names=("rr", "cc")) as tp:
                    xr = tf.schedule((Xs, tf.empty(Tensor[(C, C), "f32", ((4, 16 @ tp.rr, 16 @ tp.cc, 4), (4, 0, 0, 1)), "rmem"])), op=T.copy())
                    bj = tf.schedule((tf.reshape(Bs, new_shape=(1, C)), tf.empty(Tensor[(1, C), "f32", ((1, 1, 16 @ tp.cc, 4), (0, 0, 0, 1)), "rmem"])), op=T.copy())
                    bgj = tf.schedule((BGs, tf.empty(Tensor[(1, C), "f32", ((1, 1, 16 @ tp.cc, 4), (0, 0, 0, 1)), "rmem"])), op=T.copy())
                    mb = xr * bj
                    mg = xr * bgj
                    mbh = tf.cast(mb, dtype="bf16")
                    mgh = tf.cast(mg, dtype="bf16")
                    Mbh = tf.schedule((mbh, tf.empty(Tensor[(C, C), "bf16", Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy())
                    Mbl = tf.schedule((tf.cast(mb - tf.cast(mbh, dtype="f32"), dtype="bf16"), tf.empty(Tensor[(C, C), "bf16", Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy())
                    Mgh = tf.schedule((mgh, tf.empty(Tensor[(C, C), "bf16", Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy())
                    Mgl = tf.schedule((tf.cast(mg - tf.cast(mgh, dtype="f32"), dtype="bf16"), tf.empty(Tensor[(C, C), "bf16", Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy())
                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as cw:
                    UT = tf.schedule((tf.zeros(Tensor[(D, C), "f32", ((2, 2 @ cw.group, 8 @ cw.lane8, 2, 4 @ cw.warp, 2, 4 @ cw.lane4, 4), (2048, 4096, 1, 8, 16, 64, 128, 512)), "rmem"]), tf.bitcast(vs, Layout(((2, 1, 64), (4, 2, 8)), ((4096, 1024, 1), (1024, 512, 64))) | Swizzle(3, 4, 3)), tf.bitcast(Mbh, Layout(((4, 16), (2, 4, 8)), ((16, 1), (2048, 512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=u_mma), repeat=(2, 2, 4))
                    UT = tf.schedule((UT, tf.bitcast(vs, Layout(((2, 1, 64), (4, 2, 8)), ((4096, 1024, 1), (1024, 512, 64))) | Swizzle(3, 4, 3)), tf.bitcast(Mbl, Layout(((4, 16), (2, 4, 8)), ((16, 1), (2048, 512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=u_mma), repeat=(2, 2, 4))
                    WF = tf.schedule((tf.zeros(Tensor[(C, D), "f32", ((2 @ cw.group, 8 @ cw.lane8, 2, 4 @ cw.warp, 2, 4 @ cw.lane4, 8), (4096, 1, 8, 16, 64, 128, 512)), "rmem"]), Mgh, tf.bitcast(ks, Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=w_mma), repeat=(1, 2, 4))
                    WF = tf.schedule((WF, Mgl, tf.bitcast(ks, Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=w_mma), repeat=(1, 2, 4))
                    UTs = tf.schedule((UT, tf.empty(Tensor[(D, C), "f32", Layout(((16, 8), (2, 32)), ((256, 32), (4096, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy())
                    wh = tf.cast(WF, dtype="bf16")
                    wlo = tf.cast(WF - tf.cast(wh, dtype="f32"), dtype="bf16")
                    Whs = tf.schedule((tf.cast(wh, dtype="f32"), tf.empty(Tensor[(C, D), "f32", Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy())
                    Wls = tf.schedule((tf.cast(wlo, dtype="f32"), tf.empty(Tensor[(C, D), "f32", Layout(((8, 8), (4, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy())
                with threads[0, :32] as _st:
                    uo = tf.schedule((UTs, tf.empty(Tensor[(D, C), "f32", (D, C)])), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tf.reshape(uo, new_shape=(1, 1, C * D)), (cta.h, cta.c, U_OFF))
                    who = tf.schedule((Whs, tf.empty(Tensor[(C, D), "f32", (C, D)])), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tf.reshape(who, new_shape=(1, 1, C * D)), (cta.h, cta.c, WH_OFF))
                    wlo_ = tf.schedule((Wls, tf.empty(Tensor[(C, D), "f32", (C, D)])), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tf.reshape(wlo_, new_shape=(1, 1, C * D)), (cta.h, cta.c, WL_OFF))
                return out
