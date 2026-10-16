"""Scheduled HIR: GDN scan, plain variant (sequential pass over chunks) at prompt length 256 (NC = 4).

Implements model.py Qwen3_8DeltaNetLayer.delta_rule, the pass over the chunks: the
recurrent state across chunks, the read-out o (landed in bf16) and the final state.
The gated RMSNorm of gated_delta_net is a separate kernel (scheduled/gdn/gdn_norm_T256.py).
Runtime kernel: runtime/kernels/gdn.py make_scan(256, 16, 16640, gated=False)
(`gdn_scan_plain`, the z=None path of GdnKernels.core; precise state path).

Decomposition (as the kernel): CTA mesh (value head h, dv half vb), vb fastest (tilelang
grid (2, 48) of independent CTAs, no cluster). Each CTA of the pair owns 64 of the
128 dv columns independently: CTA vb holds S^T rows [64vb, 64vb+64) (64 dv x 128 dk,
f32) in the wgmma accumulator of warpgroup 0 and writes only its own read-out columns
and state columns; no data crosses between the two CTAs. 384 threads = 3 warpgroups:
- warpgroup 0 (math), per chunk c, with bf16 hi + lo splits S = Sh + Sl, W = Wh + Wl:
    S^T *= exp(G_63)                      (F[2, 0], scaled before any wgmma)
    Y = Sh Wh^T + Sh Wl^T + Sl Wh^T       (wgmma RS m64n64k16, S from registers)
    O = Sh Q^T                            (wgmma RS m64n64k16)
    Delta^T = U^T - Y (f32), Dk = Delta^T * exp(G_63 - G_j) split bf16 hi + lo
    S^T += Dh K + Dl K                    (wgmma RS, K MN-major from the K tile)
    O = O * exp(G_j) / sqrt(128) + bf16(Delta^T) P^T   (wgmma RS m64n64k16)
    bf16(O) -> shared tile O_s[t, dv] (transposed, rows padded to 72, 2 buffers).
  The bf16 S, Delta and Dk operands are restated from the accumulator layout into the
  wgmma A-operand register layout (k-tile mode first) by register copies.
- warpgroup 1, one warp: TMA loads per chunk into 2-deep rings: W hi, W lo, Q, K
  (64 x 128 bf16, 128B-swizzled panels), P (64 x 64 bf16), U^T rows of this dv half
  (64 x 64 f32, swizzled), F (3 x 64 f32).
- warpgroups 1 + 2 (store): thread (half, r, p) copies row t = 32*half + r of O_s,
  dv columns 16p .. 16p + 15, to the read-out (64 values per row and CTA).
After the last chunk warpgroup 0 stores S^T transposed (state[h, dk, 64vb + dv]).

Differences to the kernel:
- State update instruction: the kernel issues one m64n128k16 per k16 step whose B
  descriptor spans both 64-column panels of the K tile (panel stride 4096 elements,
  LBO 8192 B). TileFoundry's MN-major B pattern requires the panel stride to be
  16 * 64 elements and refuses the kernel's tile ("rhs: could not satisfy lb == 16 *
  Wb"), so S^T += D K is stated as two m64n64k16 atoms per k16 step (one per panel,
  repeat (1, 2, 4)) on the same K tile; the accumulator layout lists the N tile mode
  first. Same products, same f32 accumulation per element.
- Pipelining: the kernel stores the read-out one chunk behind the math (store of chunk
  c-1 in iteration c) and runs the math warpgroup ahead through mbarrier-guarded
  double buffers (full / empty, ofull / oempty). This program states one chunk per
  loop iteration with 2-deep rings (buffers=2); the overlap is left to lowering.
- Register reallocation (setmaxnreg 240 / 128) and warp-specialisation barriers are
  not stated (no TileFoundry construct).
- Result: scheduled lowering accepts one tensor result. The read-out (T, 48, 128)
  bf16 and the final state (48, 128, 128) f32 are one f32 tensor (T + 128, 48, 128):
  rows 0..T-1 = read-out (bf16 values widened exactly), row T + dk = state[:, dk, :].
  The kernel writes the read-out as bf16 and the state as f32 into two buffers.
- Inputs: the kernel's workspaces as it reads them (U^T f32 (48, NC, 128, 64), W bf16
  (48, NC, 2, 64, 128), P bf16 (48, NC, 64, 64), F f32 (48, NC, 3, 64)); the
  scheduled gdn_chunk states them packed in one f32 tensor (see its header).
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

T_ = 256
NC = T_ // 64
HV = 48
D = 128
C = 64
S_ROW = T_


@module(entry="gdn_scan", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", HV * 2), Topology("thread", 384)))
class SCHED_GDN_SCAN_T256:
    @func
    def gdn_scan(q: Tensor[(T_, 16, D), "bf16"], k: Tensor[(T_, 16, D), "bf16"], ut: Tensor[(HV, NC, D, C), "f32"],
                 wb: Tensor[(HV, NC, 2, C, D), "bf16"], pb: Tensor[(HV, NC, C, C), "bf16"],
                 fc: Tensor[(HV, NC, 3, C), "f32"]) -> Tensor[(T_ + 128, HV, D), "f32"]:
        with Mesh(("cta",), layout=(HV, 2), names=("h", "vb")) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                rs_mma = T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                st_mma = T.cuda.sm90.Wgmma(n=64, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)
                out = tf.zeros(Tensor[(T_ + 128, HV, D), "f32"])
                kh = cta.h // 3
                d0 = cta.vb * 64
                with Mesh(threads[0:1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as cm:
                    St = tf.zeros(Tensor[(C, D), "f32", ((4 @ cm.warp, 2, 8 @ cm.lane8, 2, 8, 4 @ cm.lane4, 2), (16, 8, 1, 4096, 512, 128, 64)), "rmem"])
                for c in range(NC):
                    with threads[1, :32] as _ld:
                        W_s = tf.schedule((tf.reshape(wb[cta.h:cta.h + 1, c:c + 1, 0:1, :, :], new_shape=(C, D)), tf.empty(Tensor[(C, D), "bf16", Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=2)), op=T.copy_async_tensor())
                        Wl_s = tf.schedule((tf.reshape(wb[cta.h:cta.h + 1, c:c + 1, 1:2, :, :], new_shape=(C, D)), tf.empty(Tensor[(C, D), "bf16", Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=2)), op=T.copy_async_tensor())
                        Q_s = tf.schedule((tf.reshape(q[c * C:c * C + C, kh:kh + 1, :], new_shape=(C, D)), tf.empty(Tensor[(C, D), "bf16", Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=2)), op=T.copy_async_tensor())
                        K_s = tf.schedule((tf.reshape(k[c * C:c * C + C, kh:kh + 1, :], new_shape=(C, D)), tf.empty(Tensor[(C, D), "bf16", Layout(((8, 8), (2, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=2)), op=T.copy_async_tensor())
                        P_s = tf.schedule((tf.reshape(pb[cta.h:cta.h + 1, c:c + 1, :, :], new_shape=(C, C)), tf.empty(Tensor[(C, C), "bf16", Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=2)), op=T.copy_async_tensor())
                        U_s = tf.schedule((tf.reshape(ut[cta.h:cta.h + 1, c:c + 1, d0:d0 + 64, :], new_shape=(C, C)), tf.empty(Tensor[(C, C), "f32", Layout(((8, 8), (2, 32)), ((256, 32), (2048, 1))) | Swizzle(3, 4, 3), "smem"], buffers=2)), op=T.copy_async_tensor())
                        F_s = tf.schedule((tf.reshape(fc[cta.h:cta.h + 1, c:c + 1, :, :], new_shape=(3, C)), tf.empty(Tensor[(3, C), "f32", Layout((3, C), (C, 1)), "smem"], buffers=2)), op=T.copy_async_tensor())
                    with Mesh(threads[0:1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as cm:
                        sbv = tf.cast(St, dtype="bf16")
                        Sh = tf.schedule((sbv, tf.empty(Tensor[(C, D), "bf16", ((4 @ cm.warp, 2, 8 @ cm.lane8, 8, 2, 4 @ cm.lane4, 2), (16, 8, 1, 1024, 512, 128, 64)), "rmem"])), op=T.copy())
                        Sl = tf.schedule((tf.cast(St - tf.cast(sbv, dtype="f32"), dtype="bf16"), tf.empty(Tensor[(C, D), "bf16", ((4 @ cm.warp, 2, 8 @ cm.lane8, 8, 2, 4 @ cm.lane4, 2), (16, 8, 1, 1024, 512, 128, 64)), "rmem"])), op=T.copy())
                        eg = tf.schedule((F_s[2:3, 0:1], tf.empty(Tensor[(1, 1), "f32", (1, 1), "rmem"])), op=T.copy())
                        St = St * eg
                        Y = tf.schedule((tf.zeros(Tensor[(C, C), "f32", ((4 @ cm.warp, 2, 8 @ cm.lane8, 8, 4 @ cm.lane4, 2), (16, 8, 1, 512, 128, 64)), "rmem"]), Sh, tf.bitcast(W_s, Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 8))
                        Y = tf.schedule((Y, Sh, tf.bitcast(Wl_s, Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 8))
                        Y = tf.schedule((Y, Sl, tf.bitcast(W_s, Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 8))
                        O = tf.schedule((tf.zeros(Tensor[(C, C), "f32", ((4 @ cm.warp, 2, 8 @ cm.lane8, 8, 4 @ cm.lane4, 2), (16, 8, 1, 512, 128, 64)), "rmem"]), Sh, tf.bitcast(Q_s, Layout(((2, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 8))
                        u = tf.schedule((U_s, tf.empty(Tensor[(C, C), "f32", ((4 @ cm.warp, 2, 8 @ cm.lane8, 8, 4 @ cm.lane4, 2), (16, 8, 1, 512, 128, 64)), "rmem"])), op=T.copy())
                        f0 = tf.schedule((F_s[0:1, :], tf.empty(Tensor[(1, C), "f32", ((1, 8, 4 @ cm.lane4, 2, 1, 1), (0, 512, 128, 64, 0, 0)), "rmem"])), op=T.copy())
                        f1 = tf.schedule((F_s[1:2, :], tf.empty(Tensor[(1, C), "f32", ((1, 8, 4 @ cm.lane4, 2, 1, 1), (0, 512, 128, 64, 0, 0)), "rmem"])), op=T.copy())
                        dlt = u - Y
                        dk = dlt * f0
                        dkh = tf.cast(dk, dtype="bf16")
                        Dt = tf.schedule((tf.cast(dlt, dtype="bf16"), tf.empty(Tensor[(C, C), "bf16", ((4 @ cm.warp, 2, 8 @ cm.lane8, 4, 2, 4 @ cm.lane4, 2), (16, 8, 1, 1024, 512, 128, 64)), "rmem"])), op=T.copy())
                        Dk = tf.schedule((dkh, tf.empty(Tensor[(C, C), "bf16", ((4 @ cm.warp, 2, 8 @ cm.lane8, 4, 2, 4 @ cm.lane4, 2), (16, 8, 1, 1024, 512, 128, 64)), "rmem"])), op=T.copy())
                        Dkl = tf.schedule((tf.cast(dk - tf.cast(dkh, dtype="f32"), dtype="bf16"), tf.empty(Tensor[(C, C), "bf16", ((4 @ cm.warp, 2, 8 @ cm.lane8, 4, 2, 4 @ cm.lane4, 2), (16, 8, 1, 1024, 512, 128, 64)), "rmem"])), op=T.copy())
                        St = tf.schedule((St, Dk, tf.bitcast(K_s, Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=st_mma), repeat=(1, 2, 4))
                        St = tf.schedule((St, Dkl, tf.bitcast(K_s, Layout(((4, 2, 8), (2, 1, 64)), ((1024, 512, 64), (4096, 1024, 1))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=st_mma), repeat=(1, 2, 4))
                        O = tf.schedule((O * f1, Dt, tf.bitcast(P_s, Layout(((4, 16), (8, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=rs_mma), repeat=(1, 1, 4))
                        O_s = tf.schedule((tf.cast(O, dtype="bf16"), tf.empty(Tensor[(C, C), "bf16", Layout((C, C), (1, 72)), "smem"], buffers=2)), op=T.copy())
                    with Mesh(threads[1:3, :], layout=(2, 32, 4), names=("half", "r", "p")) as ce:
                        ov = tf.schedule((tf.bitcast(O_s, Layout((C, C), (72, 1))), tf.empty(Tensor[(C, C), "bf16", ((2 @ ce.half, 32 @ ce.r, 4 @ ce.p, 16), (0, 0, 0, 1)), "rmem"])), op=T.copy())
                        yo = tf.schedule((tf.cast(ov, dtype="f32"), tf.empty(Tensor[(C, C), "f32", (C, C)])), op=T.copy())
                        out = tf.insert_slice(out, tf.reshape(yo, new_shape=(C, 1, C)), (c * C, cta.h, d0))
                with Mesh(threads[0:1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as cm:
                    sto = tf.schedule((St, tf.empty(Tensor[(C, D), "f32", Layout((C, D), (1, C))])), op=T.copy())
                    out = tf.insert_slice(out, tf.reshape(tf.bitcast(sto, Layout((D, C), (C, 1))), new_shape=(D, 1, C)), (S_ROW, cta.h, d0))
                return out
