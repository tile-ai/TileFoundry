"""Scheduled HIR: attention core at prompt length 128 (heads per CTA 1, gate "early", DSPLIT 2).

Implements model.py Qwen3_8AttentionLayer.gated_attention from the core on:
  y = bf16(bf16(attend(q, k, v)) * bf16(sigmoid(gate))) for the 24 query heads, causal, GQA 6:1.
Runtime kernel: runtime/kernels/attention.py make_core(128, 1, "early", DSPLIT=2) (`attn_core`),
configuration from attention.CORE_CONFIG[128].

Decomposition (as the kernel): CTA mesh (token block, query-head group, output-dim half), the head
axis fastest and the token block walked from the last (longest causal range) down
(LPT order): bx = NB - 1 - blk. Warpgroup 0 is the producer (warp 0: Q tiles and the
K ring, warp 1: the V ring, TMA into 128B-swizzled tiles, ST = 2 deep); warpgroup
1 + w computes query head hg + w with wgmma: S = Q K^T (SS, n = 64, K-major),
online softmax in base 2 (scale 1/16 folded in c = log2(e)/16), O = O * alpha + bf16(P) V
(RS, n = 128, V MN-major), P landed in bf16 from the S accumulator registers.
The key blocks below the diagonal run in the loop; the diagonal block (the only one that
crosses it, TM = BN = 64) is peeled after the loop and gets the causal mask. The mask
tile is a constant input `diag` (0 on/below the diagonal, -inf above); the kernel
evaluates the same predicate from indices (T.if_then_else(..., S, -inf) == S + diag).
Gate "early": warp 1 TMA-loads the gate tile before the main loop and the producer warps
(all 4) turn it into bf16(sigmoid) in shared memory while the consumer runs; the
epilogue multiplies O / l by it and stages y for one TMA store. (The kernel computes
the sigmoid in place in the gate buffer; this program names a second buffer.)
Rounding (HIR attend): f32 scores, running max and sum in f32, P bf16 for the PV wgmma,
f32 accumulation, O / l landed in bf16, gate product in f32 landed in bf16.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

T_ = 128
D = 256
DV = 128
TM = 64
BN = 64
ST = 2
NB = T_ // TM
NQ = 24
NG = 24
LOG2E = 1.4426950408889634
NLOG2E = -1.4426950408889634
C = LOG2E / 16.0
NEG = float("-inf")
QKV = 14336


@module(entry="attn_core", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", NG * NB * 2), Topology("thread", 256)))
class SCHED_ATTN_CORE_T128:
    @func
    def attn_core(q: Tensor[(T_, NQ, D), "bf16"], k: Tensor[(T_, 4, D), "bf16"], v: Tensor[(T_, 4, D), "bf16"],
                  qkv: Tensor[(T_, QKV), "bf16"], diag: Tensor[(TM, BN), "f32"]) -> Tensor[(T_, NQ * D), "bf16"]:
        with Mesh(("cta",), layout=(NB, NG, 2), names=("blk", "hg", "dh")) as cta:
            with Mesh(("thread",), layout=(2, 128), names=("warpgroup", "participant")) as threads:
                s_mma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                o_mma = T.cuda.sm90.Wgmma(n=DV, dtype="bf16", form=T.cuda.sm90.Form.RS, a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN)
                out = tf.zeros(Tensor[(T_, NQ * D), "bf16"])
                bx = NB - 1 - cta.blk
                g = cta.hg // 6
                with threads[0, :32] as _ql:
                    q_dst0 = tf.empty(Tensor[(TM, D), "bf16", Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"])
                    qt0 = tf.schedule((tf.reshape(q[bx * TM:bx * TM + TM, cta.hg:cta.hg + 1, :], new_shape=(TM, D)), q_dst0), op=T.copy_async_tensor())
                with threads[0, 32:64] as _gl:
                    g_dst = tf.empty(Tensor[(TM, DV), "bf16", (TM, DV), "smem"])
                    gt = tf.schedule((qkv[bx * TM:bx * TM + TM, (cta.hg) * 512 + D + cta.dh * DV:(cta.hg) * 512 + D + cta.dh * DV + DV], g_dst), op=T.copy_async_tensor())
                with Mesh(threads[0, :], layout=(64, 2), names=("r", "h")) as _s:
                    gr = tf.schedule((gt, tf.empty(Tensor[(TM, DV), "bf16", ((64 @ _s.r, 2 @ _s.h, DV // 2), (0, 0, 1)), "rmem"])), op=T.copy())
                    sig = tf.cast(1.0 / (tf.exp2(tf.cast(gr, dtype="f32") * NLOG2E) + 1.0), dtype="bf16")
                    gs = tf.schedule((sig, tf.empty(Tensor[(TM, DV), "bf16", (TM, DV), "smem"])), op=T.copy())
                with Mesh(threads[1:2, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _c0:
                    O0 = tf.zeros(Tensor[(TM, DV), "f32", ((8 @ _c0.lane8, 2, 4 @ _c0.warp, 2, 4 @ _c0.lane4, 16), (1, 8, 16, 64, 128, 512)), "rmem"])
                    S0_0 = tf.zeros(Tensor[(TM, BN), "f32", ((8 @ _c0.lane8, 2, 4 @ _c0.warp, 2, 4 @ _c0.lane4, 8), (1, 8, 16, 64, 128, 512)), "rmem"])
                    m0 = tf.reduce(S0_0, axes=(-1,), keepdim=True, kind="max") + NEG
                    l0 = tf.reduce(S0_0, axes=(-1,), keepdim=True, kind="sum")
                    dmask0 = tf.schedule((diag, tf.empty(Tensor[(TM, BN), "f32", ((8 @ _c0.lane8, 2, 4 @ _c0.warp, 2, 4 @ _c0.lane4, 8), (1, 8, 16, 64, 128, 512)), "rmem"])), op=T.copy())
                for j in range(NB - 1 - cta.blk):
                    with threads[0, :32] as _kl:
                        k_ring = tf.empty(Tensor[(BN, D), "bf16", Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=ST)
                        kt = tf.schedule((tf.reshape(k[j * BN:j * BN + BN, g:g + 1, :], new_shape=(BN, D)), k_ring), op=T.copy_async_tensor())
                    with threads[0, 32:64] as _vl:
                        v_ring = tf.empty(Tensor[(BN, DV), "bf16", Layout(((4, 2, 8), (2, 64)), ((2048, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), "smem"], buffers=ST)
                        vt = tf.schedule((tf.reshape(v[j * BN:j * BN + BN, g:g + 1, cta.dh * DV:cta.dh * DV + DV], new_shape=(BN, DV)), v_ring), op=T.copy_async_tensor())
                    with Mesh(threads[1:2, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _c0:
                        S0 = tf.schedule((tf.zeros(Tensor[(TM, BN), "f32", ((8 @ _c0.lane8, 2, 4 @ _c0.warp, 2, 4 @ _c0.lane4, 8), (1, 8, 16, 64, 128, 512)), "rmem"]), qt0, tf.bitcast(kt, Layout(((4, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=s_mma), repeat=(1, 1, 16))
                        m_new0 = tf.max(m0, tf.reduce(S0, axes=(-1,), keepdim=True, kind="max"))
                        alpha0 = tf.exp2((m0 - m_new0) * C)
                        P0 = tf.exp2(S0 * C - m_new0 * C)
                        l0 = l0 * alpha0 + tf.reduce(P0, axes=(-1,), keepdim=True, kind="sum")
                        Pb0 = tf.schedule((tf.cast(P0, dtype="bf16"), tf.empty(Tensor[(TM, BN), "bf16", ((4, 8 @ _c0.lane8, 2, 4 @ _c0.warp, 2, 4 @ _c0.lane4, 2), (1024, 1, 8, 16, 64, 128, 512)), "rmem"])), op=T.copy())
                        O0 = tf.schedule((O0 * alpha0, Pb0, vt), op=T.tiled_mma(atom=o_mma), repeat=(1, 1, 4))
                        m0 = m_new0
                with threads[0, :32] as _kl:
                    k_ring = tf.empty(Tensor[(BN, D), "bf16", Layout(((8, 8), (4, 4, 16)), ((512, 64), (4096, 16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=ST)
                    kt = tf.schedule((tf.reshape(k[bx * BN:bx * BN + BN, g:g + 1, :], new_shape=(BN, D)), k_ring), op=T.copy_async_tensor())
                with threads[0, 32:64] as _vl:
                    v_ring = tf.empty(Tensor[(BN, DV), "bf16", Layout(((4, 2, 8), (2, 64)), ((2048, 512, 64), (1024, 1))) | Swizzle(3, 4, 3), "smem"], buffers=ST)
                    vt = tf.schedule((tf.reshape(v[bx * BN:bx * BN + BN, g:g + 1, cta.dh * DV:cta.dh * DV + DV], new_shape=(BN, DV)), v_ring), op=T.copy_async_tensor())
                with Mesh(threads[1:2, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _c0:
                    S0 = tf.schedule((tf.zeros(Tensor[(TM, BN), "f32", ((8 @ _c0.lane8, 2, 4 @ _c0.warp, 2, 4 @ _c0.lane4, 8), (1, 8, 16, 64, 128, 512)), "rmem"]), qt0, tf.bitcast(kt, Layout(((4, 4, 16), (8, 8)), ((4096, 16, 1), (512, 64))) | Swizzle(3, 4, 3))), op=T.tiled_mma(atom=s_mma), repeat=(1, 1, 16))
                    S0 = S0 + dmask0
                    m_new0 = tf.max(m0, tf.reduce(S0, axes=(-1,), keepdim=True, kind="max"))
                    alpha0 = tf.exp2((m0 - m_new0) * C)
                    P0 = tf.exp2(S0 * C - m_new0 * C)
                    l0 = l0 * alpha0 + tf.reduce(P0, axes=(-1,), keepdim=True, kind="sum")
                    Pb0 = tf.schedule((tf.cast(P0, dtype="bf16"), tf.empty(Tensor[(TM, BN), "bf16", ((4, 8 @ _c0.lane8, 2, 4 @ _c0.warp, 2, 4 @ _c0.lane4, 2), (1024, 1, 8, 16, 64, 128, 512)), "rmem"])), op=T.copy())
                    O0 = tf.schedule((O0 * alpha0, Pb0, vt), op=T.tiled_mma(atom=o_mma), repeat=(1, 1, 4))
                    m0 = m_new0
                with Mesh(threads[1:2, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _c0:
                    ob0 = tf.cast(O0 * (1.0 / l0), dtype="bf16")
                    gr0 = tf.schedule((gs, tf.empty(Tensor[(TM, DV), "bf16", ((8 @ _c0.lane8, 2, 4 @ _c0.warp, 2, 4 @ _c0.lane4, 16), (1, 8, 16, 64, 128, 512)), "rmem"])), op=T.copy())
                    y0 = tf.cast(tf.cast(ob0, dtype="f32") * tf.cast(gr0, dtype="f32"), dtype="bf16")
                    staged0 = tf.schedule((y0, tf.empty(Tensor[(TM, DV), "bf16", (TM, DV), "smem"])), op=T.copy())
                with threads[0, :32] as _st:
                    yt0 = tf.schedule((staged0, tf.empty(Tensor[(TM, DV), "bf16", (TM, DV)])), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, yt0, (bx * TM, (cta.hg) * D + cta.dh * DV))
                return out
