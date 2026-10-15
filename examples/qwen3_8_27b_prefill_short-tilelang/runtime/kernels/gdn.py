"""TileLang kernels for the GatedDeltaNet (GDN) token mixer of Qwen3.8-27B prefill.

Semantics follow the authored HIR in ``model.py`` (``Qwen3_8DeltaNetLayer``):
``causal_conv``, ``l2_normalise``, ``delta_rule`` and the gated RMSNorm of
``gated_delta_net``.  The two projection GEMMs are not here.

Kernels (one compiled set per prompt length T; T must be a multiple of 64):

  gdn_prep   proj -> q, k, v, g, beta, window          (bitwise equal to the HIR)
  gdn_chunk  per (64-row chunk, value head): the intra-chunk WY quantities
             U^T (f32), W (bf16 hi + lo), P and the chunk decay factors (fully parallel)
  gdn_scan_plain  per value head two CTAs (one dv half each, no cluster), warp
             specialised: the sequential pass over chunks -> bf16 read-out o (T, 48, 128)
             and the final state
  gdn_gated_norm  HIR gated RMSNorm of o with z, gamma -> y (bitwise equal to the HIR
             formula on the same o)

Production path of core(z given): gdn_chunk, gdn_scan_plain (o into a workspace),
gdn_gated_norm.  GdnKernels(T, fused_norm=True) instead runs gdn_scan with the norm fused
into its epilogue (row sums of o^2 exchanged between the two cluster CTAs); kept for
timing experiments only.

State-path precision: everything that feeds the recurrent state carries ~16 bits
(f32 U; W, S and the state update operand split in bf16 hi + lo), so the final
state matches the f32 HIR per element (|d| <= 0.009 |ref| + 0.009 RMS with >30x
margin); the read-out path (q.S, P.Delta) uses single bf16 operands.

All launches go to torch's current CUDA stream (TVM-FFI reads it through the
DLPack stream exchange), so the wrappers can be captured in a CUDA graph.
"""
import numpy as np
import torch
import tilelang
import tilelang.language as T
from tilelang.layout import make_swizzled_layout

__all__ = ["GdnKernels"]

BF16 = "bfloat16"
F32 = "float32"

HK = 16            # key heads produced by prep
HV = 48            # value heads
D = 128            # head dim (dk = dv)
CONV = 10240       # conv channels: q [0,2048) k [2048,4096) v [4096,10240)
PROJ = 16480       # real columns of the fused projection: mixed | z | b | a
Z_OFF = 10240
B_OFF = 16384
A_OFF = 16432
C = 64             # chunk length of the WY form
SCALE = D ** -0.5
NEG_LOG2E = -1.4426950408889634
SILU_MARGIN = 16   # f32-ulp guard around a bf16 rounding midpoint for the fast silu (max error 13 ulp for x >= -16)
SILU_XMIN = -16.0  # below this the exact silu is always taken

_FAST = {"tl.enable_fast_math": True}


def _bf16_round(x: float) -> float:
    """Python float -> f32 -> bf16 (round to nearest even) -> float."""
    u = int(np.array([np.float32(x)]).view(np.uint32)[0])
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return float(np.array([u], dtype=np.uint32).view(np.float32)[0])


# HIR: square_sum + full_like(square_sum, 1e-6) is a bf16 + bf16 add.
EPS_L2_BF16 = _bf16_round(1e-6)


# --------------------------------------------------------------------------------------
# Kernel A: conv + silu + l2norm, gates, window.
# --------------------------------------------------------------------------------------
# Approximate exp2 / reciprocal (MUFU, flush-to-zero) for the silu fast path only.  The
# exact fallback below keeps IEEE division and the accurate expf, so the prep kernel is
# compiled without fast-math.
_PREP_PRELUDE = r"""
__device__ __forceinline__ float gdn_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float gdn_rcp(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
"""
U32 = "uint32"


def _prep_config(TT: int):
    return (2, 4) if TT <= 256 else (4, 4)


def make_prep(TT: int, PS: int = PROJ, R: int = None, WARPS: int = None):
    """Grid (80 head columns, T / (R*WARPS)); a warp owns R consecutive rows of one
    128-channel head column, a lane owns 4 channels (read as uint32 words, written as 8-byte
    bf16 vectors).  silu takes a fast path (ex2/rcp approx) for all 4R values of a lane; one
    branch redoes exactly (x / (1 + expf(-x)), the torch formula) every value whose fast
    result lies within SILU_MARGIN f32 ulps of a bf16 rounding midpoint or whose x < SILU_XMIN;
    the fast path is never more than 13 ulps off for x >= -16, so every output is bitwise that
    of the HIR (also checked exhaustively over all bf16 conv values by test_gdn.py)."""
    if R is None:
        R, WARPS = _prep_config(TT)
    BT = R * WARPS
    assert TT % BT == 0
    NB = TT // BT
    E = 4 * R
    PW = PS // 2

    @T.prim_func
    def gdn_prep(
        proj: T.StridedTensor((TT, PROJ // 2), (PW, 1), U32),
        conv_w: T.Tensor((CONV, 4), BF16),
        a_log: T.Tensor((HV,), BF16),
        dt_bias: T.Tensor((HV,), BF16),
        q: T.Tensor((TT, HK, D), BF16),
        k: T.Tensor((TT, HK, D), BF16),
        v: T.Tensor((TT, HV, D), BF16),
        g: T.Tensor((TT, HV), F32),
        beta: T.Tensor((TT, HV), BF16),
        window: T.Tensor((CONV, 3), BF16),
        projb: T.StridedTensor((TT, PROJ), (PS, 1), BF16),
    ):
        with T.Kernel(80, NB, threads=WARPS * 32, prelude=_PREP_PRELUDE) as (hd, rb):
            tx = T.get_thread_binding()
            lane = tx % 32
            wp = tx // 32
            c0 = hd * 128 + lane * 4
            wl = T.alloc_local((16,), BF16)
            wf = T.alloc_local((16,), F32)
            xw = T.alloc_local((2 * (R + 3),), U32)
            xf = T.alloc_local((4 * (R + 3),), F32)
            cb = T.alloc_local((E,), F32)
            sv = T.alloc_local((E,), F32)
            ob = T.alloc_local((E,), BF16)
            flag = T.alloc_local((1,), "int32")
            red = T.alloc_local((R,), F32)
            for i in T.vectorized(16):
                wl[i] = conv_w[c0 + i // 4, i % 4]
            for i in T.unroll(16):
                wf[i] = T.cast(wl[i], F32)
            t0 = rb * BT + wp * R
            for r in T.unroll(R + 3):
                if t0 - 3 + r >= 0:
                    for i in T.vectorized(2):
                        xw[r * 2 + i] = proj[t0 - 3 + r, c0 // 2 + i]
                else:
                    for i in T.unroll(2):
                        xw[r * 2 + i] = T.uint32(0)
            for r in T.unroll(R + 3):
                for i in T.unroll(2):
                    xf[r * 4 + 2 * i] = T.reinterpret(T.shift_left(xw[r * 2 + i], T.uint32(16)), F32)
                    xf[r * 4 + 2 * i + 1] = T.reinterpret(T.bitwise_and(xw[r * 2 + i], T.uint32(0xFFFF0000)), F32)
            # depthwise causal conv: f32 products of bf16 values are exact, accumulated in tap order
            for r in T.unroll(R):
                for ch in T.unroll(4):
                    acc = xf[r * 4 + ch] * wf[ch * 4 + 0]
                    acc = acc + xf[(r + 1) * 4 + ch] * wf[ch * 4 + 1]
                    acc = acc + xf[(r + 2) * 4 + ch] * wf[ch * 4 + 2]
                    acc = acc + xf[(r + 3) * 4 + ch] * wf[ch * 4 + 3]
                    cb[r * 4 + ch] = acc
            for e in T.vectorized(E):
                ob[e] = T.cast(cb[e], BF16)
            for e in T.unroll(E):
                cb[e] = T.cast(ob[e], F32)
            # fast silu for all E values, one branch for the rare exact recomputation:
            # low16(bits) in [0x8000 - M, 0x8000 + M]  <=>  (bits << 16) - ((0x8000 - M) << 16) <= 2M << 16
            flag[0] = 0
            for e in T.unroll(E):
                x = cb[e]
                ra = x * T.call_extern(F32, "gdn_rcp", 1.0 + T.call_extern(F32, "gdn_ex2", x * NEG_LOG2E))
                sv[e] = ra
                u = T.reinterpret(ra, U32) * T.uint32(65536) - T.uint32((0x8000 - SILU_MARGIN) << 16)
                flag[0] = flag[0] | T.cast((u <= T.uint32((2 * SILU_MARGIN) << 16)) or (x < SILU_XMIN), "int32")
            if flag[0] != 0:
                for e in T.unroll(E):
                    x = cb[e]
                    u = T.reinterpret(sv[e], U32) * T.uint32(65536) - T.uint32((0x8000 - SILU_MARGIN) << 16)
                    if (u <= T.uint32((2 * SILU_MARGIN) << 16)) or (x < SILU_XMIN):
                        sv[e] = x / (1.0 + T.exp(-x))
            for e in T.vectorized(E):
                ob[e] = T.cast(sv[e], BF16)
            if hd < 32:
                # l2 normalise in bf16 arithmetic: s = bf16(sum bf16(x*x)); r = bf16(rsqrt(bf16(s + eps)))
                for e in T.unroll(E):
                    sv[e] = T.cast(ob[e], F32)
                for e in T.unroll(E):
                    cb[e] = sv[e] * sv[e]
                for e in T.vectorized(E):
                    ob[e] = T.cast(cb[e], BF16)
                for r in T.unroll(R):
                    red[r] = T.cast(ob[r * 4], F32)
                    for ch in T.unroll(1, 4):
                        red[r] = red[r] + T.cast(ob[r * 4 + ch], F32)
                for o in T.unroll(5):
                    for r in T.unroll(R):
                        red[r] = red[r] + T.shfl_xor(red[r], 16 >> o)
                for r in T.unroll(R):
                    s = T.cast(T.cast(red[r], BF16), F32)
                    se = T.cast(T.cast(s + EPS_L2_BF16, BF16), F32)
                    rr = T.cast(T.cast(T.rsqrt(se), BF16), F32)
                    for ch in T.unroll(4):
                        cb[r * 4 + ch] = sv[r * 4 + ch] * rr
                for e in T.vectorized(E):
                    ob[e] = T.cast(cb[e], BF16)
            if hd < 16:
                for r in T.unroll(R):
                    for i in T.vectorized(4):
                        q[t0 + r, hd, lane * 4 + i] = ob[r * 4 + i]
            elif hd < 32:
                for r in T.unroll(R):
                    for i in T.vectorized(4):
                        k[t0 + r, hd - 16, lane * 4 + i] = ob[r * 4 + i]
            else:
                for r in T.unroll(R):
                    for i in T.vectorized(4):
                        v[t0 + r, hd - 32, lane * 4 + i] = ob[r * 4 + i]
                if tx < BT:
                    h = hd - 32
                    t = rb * BT + tx
                    av = T.cast(projb[t, A_OFF + h], F32) + T.cast(dt_bias[h], F32)
                    sp = T.if_then_else(av > 20.0, av, T.log1p(T.exp(av)))
                    g[t, h] = -T.exp(T.cast(a_log[h], F32)) * sp
                    bv = T.cast(projb[t, B_OFF + h], F32)
                    beta[t, h] = T.cast(1.0 / (1.0 + T.exp(-bv)), BF16)
            if rb == NB - 1:
                if tx < 128:
                    for j in T.unroll(3):
                        window[hd * 128 + tx, j] = projb[TT - 3 + j, hd * 128 + tx]

    return gdn_prep


# --------------------------------------------------------------------------------------
# Kernel B1: intra-chunk quantities, one CTA per (chunk, value head).
# --------------------------------------------------------------------------------------
def make_chunk(TT: int, KH: int = HK, NT: int = 256):
    """Per chunk c and value head h (key head h*KH//48), with G = cumsum(g) in the chunk:
        A   = strict_lower(beta_i * exp(G_i - G_j) * k_i.k_j)
        X   = (I + A)^-1     (f32: 16x16 diagonal blocks by forward substitution, then two
                              doubling levels 16 -> 32 -> 64, X21 = -X22 A21 X11)
        U^T = V^T (X diag(beta))^T                          (f32 out, X side split in bf16 hi+lo)
        W   = (X diag(beta e^G)) K                          (out as bf16 hi + lo, X side split too)
        P   = lower(q_i.k_j * exp(G_i - G_j)) / sqrt(128)   (bf16 out)
        F   = [exp(G_63 - G_j), exp(G_j) / sqrt(128), exp(G_63)]
    """
    NC = TT // C
    assert NT == 256
    LOG2E = 1.4426950408889634

    @T.prim_func
    def gdn_chunk(
        q: T.Tensor((TT, KH, D), BF16),
        k: T.Tensor((TT, KH, D), BF16),
        v: T.Tensor((TT, HV, D), BF16),
        g: T.Tensor((TT, HV), F32),
        beta: T.Tensor((TT, HV), BF16),
        Ut: T.Tensor((HV, NC, D, C), F32),
        Wb: T.Tensor((HV, NC, 2, C, D), BF16),
        Pb: T.Tensor((HV, NC, C, C), BF16),
        Fc: T.Tensor((HV, NC, 3, C), F32),
    ):
        with T.Kernel(NC, HV, threads=NT) as (c, h):
            kh = h * KH // HV
            tx = T.get_thread_binding()
            K_s = T.alloc_shared((C, D), BF16)
            Q_s = T.alloc_shared((C, D), BF16)
            V_s = T.alloc_shared((C, D), BF16)
            G_s = T.alloc_shared((C,), F32)
            B_s = T.alloc_shared((C,), F32)
            BG_s = T.alloc_shared((C,), F32)
            F_s = T.alloc_shared((3, C), F32)
            A_s = T.alloc_shared((C, 68), F32)     # padded rows: conflict-free column walks
            X_s = T.alloc_shared((C, 68), F32)
            Y_s = T.alloc_shared((C, 68), F32)
            Mbh = T.alloc_shared((C, C), BF16)
            Mbl = T.alloc_shared((C, C), BF16)
            Mgh = T.alloc_shared((C, C), BF16)
            Mgl = T.alloc_shared((C, C), BF16)
            KK = T.alloc_fragment((C, C), F32)
            QK = T.alloc_fragment((C, C), F32)
            PF = T.alloc_fragment((C, C), BF16)
            UT = T.alloc_fragment((D, C), F32)
            UT_s = T.alloc_shared((D, C), F32)
            W_s = T.alloc_shared((C, D), BF16)
            Wl_s = T.alloc_shared((C, D), BF16)
            P_s = T.alloc_shared((C, C), BF16)
            WF = T.alloc_fragment((C, D), F32)
            gl = T.alloc_local((2,), F32)
            xl = T.alloc_local((16,), F32)
            acc4 = T.alloc_local((4,), F32)
            T.annotate_layout({UT_s: make_swizzled_layout(UT_s), W_s: make_swizzled_layout(W_s),
                               Wl_s: make_swizzled_layout(Wl_s), P_s: make_swizzled_layout(P_s)})
            ld = T.alloc_barrier([1, 1])
            T.tma_copy(k[c * C:(c + 1) * C, kh, :], K_s, barrier=ld[0])
            T.tma_copy(q[c * C:(c + 1) * C, kh, :], Q_s, barrier=ld[0])
            T.tma_copy(v[c * C:(c + 1) * C, h, :], V_s, barrier=ld[1])
            if tx == 0:
                T.mbarrier_arrive(ld[0])
                T.mbarrier_arrive(ld[1])
            T.clear(X_s)
            if tx < 32:
                # inclusive prefix sum of g over the chunk (lane owns rows 2l, 2l+1), kept in log2 units
                gl[0] = g[c * C + 2 * tx, h]
                gl[1] = gl[0] + g[c * C + 2 * tx + 1, h]
                for o in T.unroll(5):
                    sh = T.shfl_up(gl[1], 1 << o)
                    if tx >= (1 << o):
                        gl[0] = gl[0] + sh
                        gl[1] = gl[1] + sh
                G_s[2 * tx] = gl[0] * LOG2E
                G_s[2 * tx + 1] = gl[1] * LOG2E
            elif tx < 96:
                B_s[tx - 32] = T.cast(beta[c * C + tx - 32, h], F32)
            T.mbarrier_wait_parity(ld[0], 0)
            T.wgmma_gemm(K_s, K_s, KK, transpose_B=True, clear_accum=True)
            T.wgmma_gemm(Q_s, K_s, QK, transpose_B=True, clear_accum=True)
            T.wait_wgmma(0)
            T.sync_threads()
            if tx < C:
                eg = T.exp2(G_s[tx])
                BG_s[tx] = B_s[tx] * eg
                F_s[0, tx] = T.exp2(G_s[C - 1] - G_s[tx])
                F_s[1, tx] = eg * SCALE
                F_s[2, tx] = T.exp2(G_s[C - 1])
            for i, j in T.Parallel(C, C):
                dec = T.exp2(T.min(G_s[i] - G_s[j], 0.0))
                A_s[i, j] = T.if_then_else(i > j, B_s[i] * dec * KK[i, j], 0.0)
                PF[i, j] = T.cast(T.if_then_else(i >= j, QK[i, j] * dec * SCALE, 0.0), BF16)
            T.copy(PF, P_s)
            T.copy(P_s, Pb[h, c, :, :])
            T.copy(F_s, Fc[h, c, :, :])
            T.sync_threads()
            # (I + A)^-1, 16x16 diagonal blocks: thread = (block, column), forward substitution
            if tx < C:
                b = tx // 16
                cc = tx % 16
                for i in T.unroll(16):
                    acc = T.alloc_var(F32)
                    acc = T.if_then_else(i == cc, 1.0, 0.0)
                    for j in T.unroll(16):
                        if j < i:
                            acc = acc - A_s[16 * b + i, 16 * b + j] * xl[j]
                    xl[i] = acc
                for i in T.unroll(16):
                    X_s[16 * b + i, 16 * b + cc] = xl[i]
            T.sync_threads()
            # level 1 (16 -> 32): X_10 = -D_1 A_10 D_0 and X_32 = -D_3 A_32 D_2.
            # thread = (block pair, row, column pair), 1x2 register tile
            bj1 = (tx // 128) * 2
            r1 = (tx % 128) // 8
            cp1 = (tx % 8) * 2
            acc4[0] = 0.0
            acc4[1] = 0.0
            for kk in T.unroll(16):
                a = A_s[16 * bj1 + 16 + r1, 16 * bj1 + kk]
                acc4[0] = acc4[0] + a * X_s[16 * bj1 + kk, 16 * bj1 + cp1]
                acc4[1] = acc4[1] + a * X_s[16 * bj1 + kk, 16 * bj1 + cp1 + 1]
            Y_s[16 * bj1 + 16 + r1, 16 * bj1 + cp1] = acc4[0]
            Y_s[16 * bj1 + 16 + r1, 16 * bj1 + cp1 + 1] = acc4[1]
            T.sync_threads()
            acc4[0] = 0.0
            acc4[1] = 0.0
            for m in T.unroll(16):
                xm = X_s[16 * bj1 + 16 + r1, 16 * bj1 + 16 + m]
                acc4[0] = acc4[0] - xm * Y_s[16 * bj1 + 16 + m, 16 * bj1 + cp1]
                acc4[1] = acc4[1] - xm * Y_s[16 * bj1 + 16 + m, 16 * bj1 + cp1 + 1]
            X_s[16 * bj1 + 16 + r1, 16 * bj1 + cp1] = acc4[0]
            X_s[16 * bj1 + 16 + r1, 16 * bj1 + cp1 + 1] = acc4[1]
            T.sync_threads()
            # level 2 (32 -> 64): X[32:64, 0:32] = -X[32:64, 32:64] (A[32:64, 0:32] X[0:32, 0:32]),
            # 2x2 register tiles, uniform code in both halves (the cleared upper blocks of X_s add
            # exact zeros); a branch per half would put ThreadSync's barrier inside the branch
            hw = tx // 128
            rp = (tx % 128) // 8
            cq = hw * 8 + (tx % 8)
            for z in T.unroll(4):
                acc4[z] = 0.0
            for kk in T.unroll(32):
                a0 = A_s[32 + 2 * rp, kk]
                a1 = A_s[33 + 2 * rp, kk]
                x0 = X_s[kk, 2 * cq]
                x1 = X_s[kk, 2 * cq + 1]
                acc4[0] = acc4[0] + a0 * x0
                acc4[1] = acc4[1] + a0 * x1
                acc4[2] = acc4[2] + a1 * x0
                acc4[3] = acc4[3] + a1 * x1
            Y_s[32 + 2 * rp, 2 * cq] = acc4[0]
            Y_s[32 + 2 * rp, 2 * cq + 1] = acc4[1]
            Y_s[33 + 2 * rp, 2 * cq] = acc4[2]
            Y_s[33 + 2 * rp, 2 * cq + 1] = acc4[3]
            T.sync_threads()
            rq = 24 - 8 * hw + (tx % 128) // 16
            cr = (tx % 128) % 16
            for z in T.unroll(4):
                acc4[z] = 0.0
            for m in T.unroll(32):
                a0 = X_s[2 * rq, 32 + m]
                a1 = X_s[2 * rq + 1, 32 + m]
                y0 = Y_s[32 + m, 2 * cr]
                y1 = Y_s[32 + m, 2 * cr + 1]
                acc4[0] = acc4[0] - a0 * y0
                acc4[1] = acc4[1] - a0 * y1
                acc4[2] = acc4[2] - a1 * y0
                acc4[3] = acc4[3] - a1 * y1
            X_s[2 * rq, 2 * cr] = acc4[0]
            X_s[2 * rq, 2 * cr + 1] = acc4[1]
            X_s[2 * rq + 1, 2 * cr] = acc4[2]
            X_s[2 * rq + 1, 2 * cr + 1] = acc4[3]
            T.sync_threads()
            for i, j in T.Parallel(C, C):
                mb = X_s[i, j] * B_s[j]
                mg = X_s[i, j] * BG_s[j]
                Mbh[i, j] = T.cast(mb, BF16)
                Mbl[i, j] = T.cast(mb - T.cast(T.cast(mb, BF16), F32), BF16)
                Mgh[i, j] = T.cast(mg, BF16)
                Mgl[i, j] = T.cast(mg - T.cast(T.cast(mg, BF16), F32), BF16)
            T.mbarrier_wait_parity(ld[1], 0)
            T.wgmma_gemm(V_s, Mbh, UT, transpose_A=True, transpose_B=True, clear_accum=True)
            T.wgmma_gemm(V_s, Mbl, UT, transpose_A=True, transpose_B=True)
            T.wgmma_gemm(Mgh, K_s, WF, clear_accum=True)
            T.wgmma_gemm(Mgl, K_s, WF)
            T.wait_wgmma(0)
            T.copy(UT, UT_s)
            for i, j in T.Parallel(C, D):
                W_s[i, j] = T.cast(WF[i, j], BF16)
                Wl_s[i, j] = T.cast(WF[i, j] - T.cast(T.cast(WF[i, j], BF16), F32), BF16)
            T.copy(UT_s, Ut[h, c, :, :])
            T.copy(W_s, Wb[h, c, 0, :, :])
            T.copy(Wl_s, Wb[h, c, 1, :, :])

    return gdn_chunk


# --------------------------------------------------------------------------------------
# Kernel B2: sequential pass over chunks + gated RMSNorm, one 2-CTA cluster per value head.
# --------------------------------------------------------------------------------------
def make_scan(TT: int, KH: int = HK, PS: int = PROJ, gated: bool = True):
    """Two CTAs per value head h, CTA vb owns dv rows [64vb, 64vb+64) (gated: a 2-CTA cluster,
    plain: independent CTAs).  384 threads, warp specialised:

      warpgroup 0 (math): S^T (dv x dk) lives in the wgmma accumulator.  Per chunk, with
        S = Sh + Sl, W = Wh + Wl, D = Delta^T e^{G_63 - G} = Dh + Dl (bf16 hi + lo):
          Y   = Sh Wh^T + Sh Wl^T + Sl Wh^T       Delta^T = U^T (f32) - Y
          O   = (Sh Q^T) e^G/sqrt(128) + bf16(Delta^T) P^T
          S^T = e^{G_63} S^T + Dh K + Dl K
        then bf16(O) goes to a double-buffered smem tile for the epilogue.
      warpgroup 1: TMA producer (2-stage ring of W, q, k, P, U^T, F, and the z tiles)
        and epilogue for rows 0..31 of each chunk.
      warpgroup 2: epilogue for rows 32..63.
    Epilogue (gated): per row t the sum of squares of its 64 values is exchanged with the
    peer CTA through DSMEM (bulk copy + remote mbarrier), then
        n = bf16(o * rsqrt(mean(o^2) + 1e-6)); y = bf16(f32(bf16(n * gamma)) * silu(z)).
    Phase A (sums, send) of chunk c runs before phase B (norm, gate, store) of chunk c-1,
    so the exchange latency is hidden.  Plain (gated=False): y = bf16(o)."""
    NC = TT // C

    @T.macro
    def phase_a(c, O_s, part_s, rem_s, remb, rank, r, p, ov, ssq, r0, wgbar):
        b = c % 2
        x3 = c % 3
        ssq[0] = 0.0
        for gi in T.unroll(2):
            for e in T.vectorized(8):
                ov[e] = O_s[b, r, p * 16 + gi * 8 + e]
            for e in T.unroll(8):
                ssq[0] = ssq[0] + T.cast(ov[e], F32) * T.cast(ov[e], F32)
        ssq[0] = ssq[0] + T.shfl_xor(ssq[0], 1)
        ssq[0] = ssq[0] + T.shfl_xor(ssq[0], 2)
        if p == 0:
            part_s[x3, r] = ssq[0]
        T.sync_threads(wgbar, 128)
        T.fence_proxy_async()
        T.copy_cluster(part_s[x3, r0:r0 + 32], rem_s[x3, r0:r0 + 32], dst_block=1 - rank, remote_barrier=remb[x3])

    @T.macro
    def phase_b(cp, O_s, Zr, gam_s, part_s, rem_s, remb, zfull, y, h, vb, r, p, ov, zv, gv, res):
        bp = cp % 2
        if gated:
            xp = cp % 3
            T.mbarrier_wait_parity(remb[xp], (cp // 3) % 2)
            tot = part_s[xp, r] + rem_s[xp, r]
            rr = T.rsqrt(tot * (1.0 / 128.0) + 1e-6)
            T.mbarrier_wait_parity(zfull[bp], (cp // 2) % 2)
            for gi in T.unroll(2):
                for e in T.vectorized(8):
                    ov[e] = O_s[bp, r, p * 16 + gi * 8 + e]
                for e in T.vectorized(8):
                    zv[e] = Zr[bp, r, p * 16 + gi * 8 + e]
                for e in T.vectorized(8):
                    gv[e] = gam_s[p * 16 + gi * 8 + e]
                for e in T.unroll(8):
                    nb = T.cast(T.cast(T.cast(ov[e], F32) * rr, BF16), F32)
                    n2 = T.cast(T.cast(nb * T.cast(gv[e], F32), BF16), F32)
                    zf = T.cast(zv[e], F32)
                    res[e] = n2 * (zf * T.fast_rcp(1.0 + T.exp2(zf * NEG_LOG2E)))
                for e in T.unroll(8):
                    ov[e] = T.cast(res[e], BF16)
                for e in T.vectorized(8):
                    y[cp * C + r, h, vb * 64 + p * 16 + gi * 8 + e] = ov[e]
        else:
            for gi in T.unroll(2):
                for e in T.vectorized(8):
                    ov[e] = O_s[bp, r, p * 16 + gi * 8 + e]
                for e in T.vectorized(8):
                    y[cp * C + r, h, vb * 64 + p * 16 + gi * 8 + e] = ov[e]

    @T.macro
    def load_stage(c, s, q, k, Ut, Wb, Pb, Fc, W_s, Wl_s, Q_s, K_s, P_s, U_s, F_s, full, h, kh, vb, l):
        T.tma_copy(Wb[h, c, 0, :, :], W_s[s, :, :], barrier=full[s])
        T.tma_copy(Wb[h, c, 1, :, :], Wl_s[s, :, :], barrier=full[s])
        T.tma_copy(q[c * C:(c + 1) * C, kh, :], Q_s[s, :, :], barrier=full[s])
        T.tma_copy(k[c * C:(c + 1) * C, kh, :], K_s[s, :, :], barrier=full[s])
        T.tma_copy(Pb[h, c, :, :], P_s[s, :, :], barrier=full[s])
        T.tma_copy(Ut[h, c, vb * 64:(vb + 1) * 64, :], U_s[s, :, :], barrier=full[s])
        T.tma_copy(Fc[h, c, :, :], F_s[s, :, :], barrier=full[s])
        if l == 0:
            T.mbarrier_arrive(full[s])

    @T.macro
    def load_z(c, s, z, Zr, zfull, h, vb, l):
        T.tma_copy(z[c * C:(c + 1) * C, h, vb * 64:(vb + 1) * 64], Zr[s, :, :], barrier=zfull[s])
        if l == 0:
            T.mbarrier_arrive(zfull[s])

    @T.macro
    def body(q, k, Ut, Wb, Pb, Fc, z, gamma, y, state, vb, h):
        kh = h * KH // HV
        tx = T.get_thread_binding()
        rank = T.block_rank_in_cluster()
        W_s = T.alloc_shared((2, C, D), BF16)
        Wl_s = T.alloc_shared((2, C, D), BF16)
        Q_s = T.alloc_shared((2, C, D), BF16)
        K_s = T.alloc_shared((2, C, D), BF16)
        P_s = T.alloc_shared((2, C, C), BF16)
        U_s = T.alloc_shared((2, 64, C), F32)
        F_s = T.alloc_shared((2, 3, C), F32)
        Zr = T.alloc_shared((2, C, 64), BF16)
        O_s = T.alloc_shared((2, C, 72), BF16)      # padded rows: conflict-free transposed stores
        part_s = T.alloc_shared((3, C), F32)
        rem_s = T.alloc_shared((3, C), F32)
        gam_s = T.alloc_shared((64,), BF16)
        full = T.alloc_barrier([1, 1])
        empty = T.alloc_barrier([128, 128])
        ofull = T.alloc_barrier([128, 128])
        oempty = T.alloc_barrier([256, 256])
        zfull = T.alloc_barrier([1, 1])
        zempty = T.alloc_barrier([256, 256])
        remb = T.alloc_cluster_barrier([2, 2, 2])
        St = T.alloc_fragment((64, D), F32)
        Sb = T.alloc_fragment((64, D), BF16)
        Sl = T.alloc_fragment((64, D), BF16)
        Y = T.alloc_fragment((64, C), F32)
        O = T.alloc_fragment((64, C), F32)
        Dt = T.alloc_fragment((64, C), BF16)
        Dk = T.alloc_fragment((64, C), BF16)
        Dkl = T.alloc_fragment((64, C), BF16)
        ov = T.alloc_local((8,), BF16)
        zv = T.alloc_local((8,), BF16)
        gv = T.alloc_local((8,), BF16)
        res = T.alloc_local((8,), F32)
        ssq = T.alloc_local((1,), F32)
        T.annotate_layout({Zr: make_swizzled_layout(Zr), U_s: make_swizzled_layout(U_s)})
        # Two T.ws regions in sequence deadlock (a __syncthreads lands between them);
        # nesting the roles in one if/else keeps them concurrent.
        if tx < 128:
            with T.ws(0):
                T.inc_max_nreg(240)
                T.clear(St)
                for c in T.serial(NC):
                    s = c % 2
                    # S in bf16 hi + lo: the state path (Y and the S update) carries ~16 bits
                    for i, j in T.Parallel(64, D):
                        Sb[i, j] = T.cast(St[i, j], BF16)
                        Sl[i, j] = T.cast(St[i, j] - T.cast(T.cast(St[i, j], BF16), F32), BF16)
                    T.mbarrier_wait_parity(full[s], (c // 2) % 2)
                    # scale S before any wgmma is in flight (otherwise ptxas injects waits
                    # and, inside this divergent region, serialises every wgmma: C7518)
                    eg = F_s[s, 2, 0]
                    for i, j in T.Parallel(64, D):
                        St[i, j] = St[i, j] * eg
                    T.wgmma_gemm(Sb, W_s[s, :, :], Y, transpose_B=True, clear_accum=True)
                    T.wgmma_gemm(Sb, Wl_s[s, :, :], Y, transpose_B=True)
                    T.wgmma_gemm(Sl, W_s[s, :, :], Y, transpose_B=True)
                    T.wgmma_gemm(Sb, Q_s[s, :, :], O, transpose_B=True, clear_accum=True)
                    T.wait_wgmma(1)
                    for i, j in T.Parallel(64, C):
                        dlt = U_s[s, i, j] - Y[i, j]
                        dk = dlt * F_s[s, 0, j]
                        Dt[i, j] = T.cast(dlt, BF16)
                        Dk[i, j] = T.cast(dk, BF16)
                        Dkl[i, j] = T.cast(dk - T.cast(T.cast(dk, BF16), F32), BF16)
                    T.wgmma_gemm(Dk, K_s[s, :, :], St)
                    T.wgmma_gemm(Dkl, K_s[s, :, :], St)
                    T.wait_wgmma(2)
                    for i, j in T.Parallel(64, C):
                        O[i, j] = O[i, j] * F_s[s, 1, j]
                    T.wgmma_gemm(Dt, P_s[s, :, :], O, transpose_B=True)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(empty[s])
                    if c >= 2:
                        T.mbarrier_wait_parity(oempty[s], ((c - 2) // 2) % 2)
                    for i, j in T.Parallel(64, C):
                        O_s[s, j, i] = T.cast(O[i, j], BF16)
                    T.mbarrier_arrive(ofull[s])
                for i, j in T.Parallel(64, D):
                    state[h, j, vb * 64 + i] = St[i, j]
        elif tx < 256:
            with T.ws(1):
                T.dec_max_nreg(128)
                l = tx - 128
                r = l // 4
                p = l % 4
                if gated:
                    if l < 64:
                        gam_s[l] = gamma[vb * 64 + l]
                for c0 in T.unroll(2):
                    if c0 < NC:
                        load_stage(c0, c0, q, k, Ut, Wb, Pb, Fc, W_s, Wl_s, Q_s, K_s, P_s, U_s, F_s, full, h, kh, vb, l)
                        if gated:
                            load_z(c0, c0, z, Zr, zfull, h, vb, l)
                for c in T.serial(NC + 1):
                    b = c % 2
                    if c + 2 < NC:
                        T.mbarrier_wait_parity(empty[b], (c // 2) % 2)
                        load_stage(c + 2, b, q, k, Ut, Wb, Pb, Fc, W_s, Wl_s, Q_s, K_s, P_s, U_s, F_s, full, h, kh, vb, l)
                    if c < NC:
                        T.mbarrier_wait_parity(ofull[b], (c // 2) % 2)
                        if gated:
                            phase_a(c, O_s, part_s, rem_s, remb, rank, r, p, ov, ssq, 0, 1)
                    if c > 0:
                        cp = c - 1
                        phase_b(cp, O_s, Zr, gam_s, part_s, rem_s, remb, zfull, y, h, vb, r, p, ov, zv, gv, res)
                        T.mbarrier_arrive(oempty[cp % 2])
                        if gated:
                            T.mbarrier_arrive(zempty[cp % 2])
                            if cp + 2 < NC:
                                T.mbarrier_wait_parity(zempty[cp % 2], (cp // 2) % 2)
                                load_z(cp + 2, cp % 2, z, Zr, zfull, h, vb, l)
        else:
            with T.ws(2):
                T.dec_max_nreg(128)
                l = tx - 256
                r = 32 + l // 4
                p = l % 4
                for c in T.serial(NC + 1):
                    b = c % 2
                    if c < NC:
                        T.mbarrier_wait_parity(ofull[b], (c // 2) % 2)
                        if gated:
                            phase_a(c, O_s, part_s, rem_s, remb, rank, r, p, ov, ssq, 32, 2)
                    if c > 0:
                        cp = c - 1
                        phase_b(cp, O_s, Zr, gam_s, part_s, rem_s, remb, zfull, y, h, vb, r, p, ov, zv, gv, res)
                        T.mbarrier_arrive(oempty[cp % 2])
                        if gated:
                            T.mbarrier_arrive(zempty[cp % 2])
        if gated:
            # no CTA may exit while its peer can still write into its shared memory
            T.cluster_arrive()
            T.cluster_wait()

    if gated:
        @T.prim_func
        def gdn_scan(
            q: T.Tensor((TT, KH, D), BF16),
            k: T.Tensor((TT, KH, D), BF16),
            Ut: T.Tensor((HV, NC, D, C), F32),
            Wb: T.Tensor((HV, NC, 2, C, D), BF16),
            Pb: T.Tensor((HV, NC, C, C), BF16),
            Fc: T.Tensor((HV, NC, 3, C), F32),
            z: T.StridedTensor((TT, HV, D), (PS, D, 1), BF16),
            gamma: T.Tensor((D,), BF16),
            y: T.Tensor((TT, HV, D), BF16),
            state: T.Tensor((HV, D, D), F32),
        ):
            with T.Kernel(2, HV, threads=384, cluster_dims=(2, 1, 1)) as (vb, h):
                body(q, k, Ut, Wb, Pb, Fc, z, gamma, y, state, vb, h)

        return gdn_scan

    @T.prim_func
    def gdn_scan_plain(
        q: T.Tensor((TT, KH, D), BF16),
        k: T.Tensor((TT, KH, D), BF16),
        Ut: T.Tensor((HV, NC, D, C), F32),
        Wb: T.Tensor((HV, NC, 2, C, D), BF16),
        Pb: T.Tensor((HV, NC, C, C), BF16),
        Fc: T.Tensor((HV, NC, 3, C), F32),
        y: T.Tensor((TT, HV, D), BF16),
        state: T.Tensor((HV, D, D), F32),
    ):
        with T.Kernel(2, HV, threads=384) as (vb, h):
            body(q, k, Ut, Wb, Pb, Fc, None, None, y, state, vb, h)

    return gdn_scan_plain


# --------------------------------------------------------------------------------------
# Kernel B3: gated RMSNorm on the plain read-out (production path).
# --------------------------------------------------------------------------------------
GATE_MARGIN = 16   # f32 ulps around a bf16 rounding midpoint of n2*silu(z) that force the exact path


def _near_mid(pf, zf):
    """1 when the f32 value pf lies within GATE_MARGIN ulps of a bf16 rounding midpoint
    (low 16 bits in [0x8000 - M, 0x8000 + M], tested as (bits << 16) - ((0x8000 - M) << 16)
    <= 2M << 16 with one multiply-add) or when z < SILU_XMIN."""
    u = T.reinterpret(pf, U32) * T.uint32(65536) - T.uint32((0x8000 - GATE_MARGIN) << 16)
    return T.bitwise_or(T.cast(u <= T.uint32((2 * GATE_MARGIN) << 16), "int32"), T.cast(zf < SILU_XMIN, "int32"))


@T.macro
def _gate_fast(n2f, zf, res, e):
    """res[e] = n2 * silu(z) with silu from ex2.approx / rcp.approx (never more than 13 f32
    ulps off for z >= SILU_XMIN)."""
    sf = zf * T.call_extern(F32, "gdn_rcp", 1.0 + T.call_extern(F32, "gdn_ex2", zf * NEG_LOG2E))
    res[e] = T.ieee_mul(n2f, sf)


@T.macro
def _gate_fix(n2f, zf, res, e):
    """Redo res[e] exactly (the HIR formula: n2 * (z / (1 + expf(-z))), IEEE f32) when the fast
    value is near a bf16 rounding midpoint.  After this, bf16(res[e]) equals the HIR result
    for every finite bf16 n2 and z (checked exhaustively by make_gate_check)."""
    if _near_mid(res[e], zf) != 0:
        res[e] = T.ieee_mul(n2f, T.ieee_fdiv(zf, T.ieee_add(T.float32(1.0), T.exp(-zf))))


def make_gated_norm(TT: int, PS: int = PROJ, RW: int = 1, WARPS: int = 4):
    """HIR ``Qwen3_5RMSNormGated`` on the bf16 read-out o, bitwise equal to model.py:
        rf = f32(o);  ms = mean(rf^2) over the 128 channels
        normed = bf16(rf * rsqrt(ms + 1e-6)) * gamma          (bf16 * bf16 -> bf16)
        y = bf16(f32(normed) * zf / (1 + exp(-zf))),  zf = f32(z)   (IEEE div, accurate expf)
    ``ms`` is summed in torch's order for ``rf.square().mean(-1)`` on a contiguous
    (..., 128) f32 tensor (ATen Reduce.cuh, input-vectorised case): 32 virtual lanes own
    4-element chunks, each chunk is ((x0^2 + x1^2) + x2^2) + x3^2, then shfl_down 16, 8,
    4, 2, 1, then * (1/128).  Here a half-warp serves one (t, head) row and a lane owns two
    chunks (8 channels, 16-byte loads): virtual offsets 16..2 are lane offsets 8..1 on each
    chunk, the last offset 1 is the add of the lane's two chunks.  A butterfly (shfl_xor)
    gives every lane the same value as lane 0 of the shfl_down tree (a + b == b + a).
    The gate n2 * silu(z) takes the fast path ``_gate_fast`` for all 8 values of a lane and
    one branch to ``_gate_fix`` when any of them is near a bf16 rounding midpoint.  All other
    f32 ops are IEEE without FMA contraction; compiled without fast-math.
    Grid: T*48 / (2*WARPS*RW) CTAs of 32*WARPS threads; each warp does RW steps of 2 rows."""
    rows = TT * HV
    per_cta = 2 * WARPS * RW
    assert rows % per_cta == 0
    NB = rows // per_cta
    INV_D = 1.0 / D
    EPS = 1e-6

    @T.prim_func
    def gdn_gated_norm(
        o: T.Tensor((TT, HV, D), BF16),
        z: T.StridedTensor((TT, HV, D), (PS, D, 1), BF16),
        gamma: T.Tensor((D,), BF16),
        y: T.Tensor((TT, HV, D), BF16),
    ):
        with T.Kernel(NB, threads=32 * WARPS, prelude=_PREP_PRELUDE) as bx:
            tx = T.get_thread_binding()
            w = tx // 32
            half = (tx % 32) // 16
            ln = tx % 16
            ol = T.alloc_local((RW * 8,), BF16)
            zl = T.alloc_local((RW * 8,), BF16)
            gl = T.alloc_local((8,), BF16)
            yl = T.alloc_local((8,), BF16)
            res = T.alloc_local((8,), F32)
            sq = T.alloc_local((2,), F32)
            flag = T.alloc_local((1,), "int32")
            for e in T.vectorized(8):
                gl[e] = gamma[ln * 8 + e]
            # all loads of the warp's RW rows first (y may alias o for the compiler)
            for r in T.unroll(RW):
                row = ((bx * WARPS + w) * RW + r) * 2 + half
                for e in T.vectorized(8):
                    ol[r * 8 + e] = o[row // HV, row % HV, ln * 8 + e]
                for e in T.vectorized(8):
                    zl[r * 8 + e] = z[row // HV, row % HV, ln * 8 + e]
            for r in T.unroll(RW):
                row = ((bx * WARPS + w) * RW + r) * 2 + half
                for p in T.unroll(2):
                    x0 = T.cast(ol[r * 8 + p * 4 + 0], F32)
                    x1 = T.cast(ol[r * 8 + p * 4 + 1], F32)
                    x2 = T.cast(ol[r * 8 + p * 4 + 2], F32)
                    x3 = T.cast(ol[r * 8 + p * 4 + 3], F32)
                    sq[p] = T.ieee_add(T.ieee_add(T.ieee_add(T.ieee_mul(x0, x0), T.ieee_mul(x1, x1)),
                                                  T.ieee_mul(x2, x2)), T.ieee_mul(x3, x3))
                for p in T.unroll(2):
                    sq[p] = T.ieee_add(sq[p], T.shfl_xor(sq[p], 8))
                    sq[p] = T.ieee_add(sq[p], T.shfl_xor(sq[p], 4))
                    sq[p] = T.ieee_add(sq[p], T.shfl_xor(sq[p], 2))
                    sq[p] = T.ieee_add(sq[p], T.shfl_xor(sq[p], 1))
                ms = T.ieee_mul(T.ieee_add(sq[0], sq[1]), T.float32(INV_D))
                rr = T.rsqrt(T.ieee_add(ms, T.float32(EPS)))
                # bf16 roundings as packed vector casts (F2FP, two lanes per instruction)
                for e in T.unroll(8):
                    res[e] = T.ieee_mul(T.cast(ol[r * 8 + e], F32), rr)
                for e in T.vectorized(8):
                    yl[e] = T.cast(res[e], BF16)
                for e in T.unroll(8):
                    res[e] = T.ieee_mul(T.cast(yl[e], F32), T.cast(gl[e], F32))
                for e in T.vectorized(8):
                    yl[e] = T.cast(res[e], BF16)
                # fast gate for the 8 values, one branch for the rare exact fix-ups
                flag[0] = 0
                for e in T.unroll(8):
                    zf = T.cast(zl[r * 8 + e], F32)
                    _gate_fast(T.cast(yl[e], F32), zf, res, e)
                    flag[0] = flag[0] | _near_mid(res[e], zf)
                if flag[0] != 0:
                    for e in T.unroll(8):
                        _gate_fix(T.cast(yl[e], F32), T.cast(zl[r * 8 + e], F32), res, e)
                for e in T.vectorized(8):
                    yl[e] = T.cast(res[e], BF16)
                for e in T.vectorized(8):
                    y[row // HV, row % HV, ln * 8 + e] = yl[e]

    return gdn_gated_norm


def make_gate_check():
    """Test kernel (not used in production): for every finite bf16 n2 and z, compares
    bf16 of ``_gate_fast`` + ``_gate_fix`` with the exact bf16(n2 * (z / (1 + expf(-z)))).  Grid 65536 (z) x 256
    threads, each thread 256 n2 values.  out[0] = mismatches, out[1] = finite pairs,
    out[2], out[3] = bits (n2, z) of one mismatch."""

    @T.prim_func
    def gdn_gate_check(out: T.Tensor((4,), "int32")):
        with T.Kernel(65536, threads=256, prelude=_PREP_PRELUDE) as zb:
            tx = T.get_thread_binding()
            res = T.alloc_local((1,), F32)
            cnt = T.alloc_local((2,), "int32")
            cnt[0] = 0
            cnt[1] = 0
            zbits = T.cast(zb, "uint16")
            zf = T.cast(T.reinterpret(zbits, BF16), F32)
            if T.bitwise_and(zb, 0x7F80) != 0x7F80:
                for i in T.serial(256):
                    nbits = tx * 256 + i
                    if T.bitwise_and(nbits, 0x7F80) != 0x7F80:
                        n2f = T.cast(T.reinterpret(T.cast(nbits, "uint16"), BF16), F32)
                        _gate_fast(n2f, zf, res, 0)
                        _gate_fix(n2f, zf, res, 0)
                        fast = T.reinterpret(T.cast(res[0], BF16), "uint16")
                        exact = T.reinterpret(T.cast(T.ieee_mul(n2f, T.ieee_fdiv(zf, T.ieee_add(T.float32(1.0), T.exp(-zf)))), BF16), "uint16")
                        cnt[1] = cnt[1] + 1
                        if fast != exact:
                            cnt[0] = cnt[0] + 1
                            out[2] = nbits
                            out[3] = zb
            T.atomic_add(out[0], cnt[0])
            T.atomic_add(out[1], cnt[1])

    return gdn_gate_check


# --------------------------------------------------------------------------------------
# Python wrapper.
# --------------------------------------------------------------------------------------
def _check(name, t, shape, dtype, strides=None):
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name}: shape {tuple(t.shape)} != {tuple(shape)}")
    if t.dtype != dtype:
        raise ValueError(f"{name}: dtype {t.dtype} != {dtype}")
    if not t.is_cuda:
        raise ValueError(f"{name}: must be a CUDA tensor")
    if strides is None:
        if not t.is_contiguous():
            raise ValueError(f"{name}: must be contiguous")
    elif tuple(t.stride()) != tuple(strides):
        raise ValueError(f"{name}: strides {tuple(t.stride())} != {tuple(strides)}")


class GdnKernels:
    """Compiled GDN mixer kernels for one prompt length.

    Args:
        T: prompt length; must be a positive multiple of 64.
        key_heads: 16 (the model: value head h reads key head h // 3) or 48 (value head h
            reads key head h; for checking ``delta_rule`` alone).  ``prep`` always writes
            16 key heads, so it needs ``key_heads == 16``.
        proj_stride: row stride (elements) of the projection buffer that ``proj`` and ``z``
            are column views of (16480 when the buffer is exactly [mixed|z|b|a], 16640 when
            the GEMM output is padded to 16640 columns).

    Accepted tensors (all CUDA; "contig" = torch-contiguous):
        prep(proj, conv_w2d, a_log, dt_bias, q, k, v, g, beta, window)
            proj     (T, 16480) bf16, strides (proj_stride, 1)  e.g. buf[:, :16480]
            conv_w2d (10240, 4) bf16 contig   (= ckpt conv_w (10240, 1, 4) squeezed)
            a_log, dt_bias (48,) bf16 contig
            q, k     (T, 16, 128) bf16 contig   (out; l2-normalised, not scaled)
            v        (T, 48, 128) bf16 contig   (out)
            g        (T, 48) f32 contig          (out)
            beta     (T, 48) bf16 contig         (out)
            window   (10240, 3) bf16 contig      (out; mixed rows T-3..T-1 transposed)
        core(q, k, v, g, beta, z, gamma, y, state)
            q, k     (T, key_heads, 128) bf16 contig
            v        (T, 48, 128) bf16 contig
            g        (T, 48) f32 contig;  beta (T, 48) bf16 contig
            z        (T, 6144) or (T, 48, 128) bf16 view with row stride proj_stride and unit
                     inner strides (e.g. buf[:, 10240:16384]); or None
            gamma    (128,) bf16 contig, or None (must be None exactly when z is None)
            y        (T, 6144) or (T, 48, 128) bf16 contig (out)
            state    (48, 128, 128) f32 contig (out; state[h, dk, dv])
        With z given, y is the gated RMSNorm output: gdn_scan_plain writes the read-out o
        into the workspace ``ws_o`` (T, 48, 128) bf16, then gdn_gated_norm writes y (bitwise
        equal to the HIR norm formula applied to o).  With z None, y = bf16(delta-rule
        read-out) laid out (T, 48, 128) -- the HIR ``delta_rule`` read with its leading 1
        dropped -- and no norm or gate is applied.
        fused_norm: False (default, production) as above; True runs the older gdn_scan with
        the norm fused in (fast silu of z, differs from the HIR formula by 1 bf16 ulp in a
        few elements), for timing experiments only.

    All outputs are written in place into caller-allocated tensors; the internal chunk
    workspaces are allocated once here, so ``prep``/``core`` can be captured in a CUDA graph.
    """

    def __init__(self, T: int, key_heads: int = HK, proj_stride: int = PROJ, device=None,
                 fused_norm: bool = False):
        if T <= 0 or T % C:
            raise ValueError(f"GdnKernels supports T that is a positive multiple of {C}, got {T}")
        if key_heads not in (16, 48):
            raise ValueError("key_heads must be 16 or 48")
        if proj_stride < PROJ:
            raise ValueError("proj_stride must be >= 16480")
        self.T = T
        self.key_heads = key_heads
        self.proj_stride = proj_stride
        self.fused_norm = bool(fused_norm)
        self.device = torch.device("cuda") if device is None else torch.device(device)
        nc = T // C
        self.prep_kernel = tilelang.compile(make_prep(T, proj_stride), target="cuda") if key_heads == HK else None
        self.chunk_kernel = tilelang.compile(make_chunk(T, key_heads), target="cuda", pass_configs=_FAST)
        self._scan_plain = None
        self._scan_fused = None
        self._gated_norm = None
        self.scan_plain_kernel()
        if key_heads == HK:          # the model path: compile what core(z=...) launches now
            if self.fused_norm:
                self.fused_scan_kernel()
            else:
                self.gated_norm_kernel()
        dev = self.device
        self.ws_ut = torch.empty(HV, nc, D, C, dtype=torch.float32, device=dev)
        self.ws_w = torch.empty(HV, nc, 2, C, D, dtype=torch.bfloat16, device=dev)
        self.ws_p = torch.empty(HV, nc, C, C, dtype=torch.bfloat16, device=dev)
        self.ws_f = torch.empty(HV, nc, 3, C, dtype=torch.float32, device=dev)
        # bf16 read-out between the plain scan and the gated norm (not used when fused_norm)
        self.ws_o = None if self.fused_norm else torch.empty(T, HV, D, dtype=torch.bfloat16, device=dev)

    def scan_plain_kernel(self):
        """gdn_scan_plain: chunked delta rule -> bf16 read-out (T, 48, 128) + final state."""
        if self._scan_plain is None:
            self._scan_plain = tilelang.compile(make_scan(self.T, self.key_heads, self.proj_stride, False),
                                                target="cuda", pass_configs=_FAST)
        return self._scan_plain

    def fused_scan_kernel(self):
        """gdn_scan with the gated RMSNorm fused in (2-CTA row-sum exchange); fused_norm=True only."""
        if self._scan_fused is None:
            self._scan_fused = tilelang.compile(make_scan(self.T, self.key_heads, self.proj_stride, True),
                                                target="cuda", pass_configs=_FAST)
        return self._scan_fused

    def gated_norm_kernel(self):
        """gdn_gated_norm: HIR gated RMSNorm of the bf16 read-out (bitwise); exact math, no fast-math."""
        if self._gated_norm is None:
            self._gated_norm = tilelang.compile(make_gated_norm(self.T, self.proj_stride), target="cuda")
        return self._gated_norm

    def prep(self, proj, conv_w2d, a_log, dt_bias, q, k, v, g, beta, window) -> None:
        if self.prep_kernel is None:
            raise RuntimeError("prep needs key_heads == 16")
        T_ = self.T
        _check("proj", proj, (T_, PROJ), torch.bfloat16, (self.proj_stride, 1))
        _check("conv_w2d", conv_w2d, (CONV, 4), torch.bfloat16)
        _check("a_log", a_log, (HV,), torch.bfloat16)
        _check("dt_bias", dt_bias, (HV,), torch.bfloat16)
        _check("q", q, (T_, HK, D), torch.bfloat16)
        _check("k", k, (T_, HK, D), torch.bfloat16)
        _check("v", v, (T_, HV, D), torch.bfloat16)
        _check("g", g, (T_, HV), torch.float32)
        _check("beta", beta, (T_, HV), torch.bfloat16)
        _check("window", window, (CONV, 3), torch.bfloat16)
        u32 = torch.uint32
        self.prep_kernel(proj.view(u32), conv_w2d, a_log, dt_bias, q, k, v, g, beta, window, proj)

    def core(self, q, k, v, g, beta, z, gamma, y, state) -> None:
        T_ = self.T
        kh = self.key_heads
        _check("q", q, (T_, kh, D), torch.bfloat16)
        _check("k", k, (T_, kh, D), torch.bfloat16)
        _check("v", v, (T_, HV, D), torch.bfloat16)
        _check("g", g, (T_, HV), torch.float32)
        _check("beta", beta, (T_, HV), torch.bfloat16)
        if y.numel() != T_ * HV * D or y.dtype != torch.bfloat16 or not y.is_contiguous():
            raise ValueError("y: needs a contiguous bf16 (T, 6144) or (T, 48, 128) tensor")
        _check("state", state, (HV, D, D), torch.float32)
        if (z is None) != (gamma is None):
            raise ValueError("z and gamma must both be given or both be None")
        y3 = y.view(T_, HV, D)
        if z is not None:
            if z.dim() == 2:
                _check("z", z, (T_, HV * D), torch.bfloat16, (self.proj_stride, 1))
                z = z.view(T_, HV, D)
            _check("z", z, (T_, HV, D), torch.bfloat16, (self.proj_stride, D, 1))
            _check("gamma", gamma, (D,), torch.bfloat16)
        ws = (self.ws_ut, self.ws_w, self.ws_p, self.ws_f)
        self.chunk_kernel(q, k, v, g, beta, *ws)
        if z is None:
            self.scan_plain_kernel()(q, k, *ws, y3, state)
        elif self.fused_norm:
            self.fused_scan_kernel()(q, k, *ws, z, gamma, y3, state)
        else:
            self.scan_plain_kernel()(q, k, *ws, self.ws_o, state)
            self.gated_norm_kernel()(self.ws_o, z, gamma, y3)
