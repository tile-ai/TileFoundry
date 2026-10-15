"""Gated full-attention token mixer of Qwen3.8-27B prefill: two tilelang kernels.

Reference: ``model.py``, class ``Qwen3_8AttentionLayer`` (``partial_rope``,
``partial_rope_kv``, ``attend``, ``gated_attention``).  See ATTENTION_NOTES.md.

Kernel A, ``attn_prep``   qkv -> q, k (RMSNorm + partial rope), v (copy).
Kernel B, ``attn_core``   causal GQA softmax attention + output gate -> y.

Tensor contract (all bf16, all on the current CUDA device, all row-major
contiguous; checked by the wrappers)::

    qkv     (T, 14336)  per row: [24 x (query 256 | gate 256)] [k 4 x 256] [v 4 x 256]
    gamma_q (256,)      gamma_k (256,)
    cos/sin (T, 64)
    q       (T, 24, 256)   written by prep, read by core
    k, v    (T, 4, 256)    written by prep (v is a copy of qkv[:, 13312:]); they may be
                           cache slices such as k_cache[i] or k_cache[i, :T] of a
                           (16, Tmax, 4, 256) cache, which are contiguous
    y       (T, 6144)      written by core: y[t, h*256 + d]

Both kernels launch on torch's current stream (TVM-FFI reads it through DLPack at
call time) and are CUDA-graph capturable.  One compiled kernel per prompt length.
"""

import torch
import tilelang
import tilelang.language as T

BF16 = "bfloat16"
F32 = "float32"

N_Q = 24            # query heads
N_KV = 4            # key/value heads
GROUP = N_Q // N_KV  # query heads per KV head
HD = 256            # head dim
ROT = 64            # rotary dims (rotate-half pairs (i, i + 32))
QKV_W = N_Q * 2 * HD + 2 * N_KV * HD   # 14336
K_COL = N_Q * 2 * HD                   # 12288
V_COL = K_COL + N_KV * HD              # 13312
Y_W = N_Q * HD                         # 6144
EPS = 1e-6
LOG2E = 1.4426950408889634
SUPPORTED_T = (128, 256, 512, 1024)


# ---------------------------------------------------------------------------
# Kernel A: attn_prep
# ---------------------------------------------------------------------------
def make_prep(T_: int):
    """One CTA per token (1024 threads, 32 warps); warp w handles slot w:
    0..23 query head w, 24..27 key head w-24, 28..31 value head w-28 (copy).

    The RMSNorm is bitwise HF/torch: lane l owns elements 4l..4l+3 and
    128+4l..128+4l+3 and sums squares in exactly the order of torch's
    ``x.pow(2).mean(-1)`` CUDA reduction (per-lane ((a0+a1)+a2)+a3 with
    a_i = x[4l+i]^2 + x[128+4l+i]^2, then shfl_down 16, 8, 4, 2, 1, times 1/256).
    Every f32 op is an explicit IEEE op (no FMA contraction).  The rope then runs
    in bf16 arithmetic exactly as the HIR: bf16(bf16(x*cos) + bf16(rot(x)*sin)).
    """

    @T.macro
    def norm_rope(qkv, cos, sin, t, lane, src_col, gamma, dst, dh):
        xi = T.alloc_local((8,), BF16)
        gi = T.alloc_local((8,), BF16)
        x = T.alloc_local((8,), F32)
        y = T.alloc_local((8,), F32)
        o = T.alloc_local((8,), BF16)
        cs = T.alloc_local((4,), BF16)
        sn = T.alloc_local((4,), BF16)
        acc = T.alloc_local((4,), F32)
        s = T.alloc_var(F32)
        r = T.alloc_var(F32)
        for i in T.vectorized(4):
            xi[i] = qkv[t, src_col + 4 * lane + i]
        for i in T.vectorized(4):
            xi[4 + i] = qkv[t, src_col + 128 + 4 * lane + i]
        for i in T.vectorized(4):
            gi[i] = gamma[4 * lane + i]
        for i in T.vectorized(4):
            gi[4 + i] = gamma[128 + 4 * lane + i]
        for i in T.vectorized(4):
            cs[i] = cos[t, (4 * lane + i) % ROT]
        for i in T.vectorized(4):
            sn[i] = sin[t, (4 * lane + i) % ROT]
        for i in T.unroll(8):
            x[i] = T.cast(xi[i], F32)
        for i in T.unroll(4):
            acc[i] = T.ieee_add(T.ieee_mul(x[i], x[i]), T.ieee_mul(x[4 + i], x[4 + i]))
        s = T.ieee_add(T.ieee_add(T.ieee_add(acc[0], acc[1]), acc[2]), acc[3])
        s = T.ieee_add(s, T.shfl_down(s, 16))
        s = T.ieee_add(s, T.shfl_down(s, 8))
        s = T.ieee_add(s, T.shfl_down(s, 4))
        s = T.ieee_add(s, T.shfl_down(s, 2))
        s = T.ieee_add(s, T.shfl_down(s, 1))
        s = T.shfl_sync(s, 0)
        r = T.rsqrt(T.ieee_add(T.ieee_mul(s, T.float32(1.0 / HD)), T.float32(EPS)))
        # x * rsqrt(ms + eps) * (1 + f32(gamma)), landed once in bf16
        for i in T.unroll(8):
            y[i] = T.cast(T.cast(T.ieee_mul(T.ieee_mul(x[i], r), T.ieee_add(T.cast(gi[i], F32), T.float32(1.0))), BF16), F32)
        # rotate-half on dims 0..63, all held by lanes 0..15 in y[0:4]; partner = lane ^ 8
        for i in T.unroll(4):
            pv = T.shfl_xor(y[i], 8)
            rotated = T.if_then_else(lane < 8, -pv, pv)
            a1 = T.cast(T.cast(T.ieee_mul(y[i], T.cast(cs[i], F32)), BF16), F32)
            a2 = T.cast(T.cast(T.ieee_mul(rotated, T.cast(sn[i], F32)), BF16), F32)
            o[i] = T.if_then_else(lane < 16, T.cast(T.ieee_add(a1, a2), BF16), T.cast(y[i], BF16))
        for i in T.unroll(4):
            o[4 + i] = T.cast(y[4 + i], BF16)
        for i in T.vectorized(4):
            dst[t, dh, 4 * lane + i] = o[i]
        for i in T.vectorized(4):
            dst[t, dh, 128 + 4 * lane + i] = o[4 + i]

    @T.prim_func
    def attn_prep(qkv: T.Tensor((T_, QKV_W), BF16), gq: T.Tensor((HD,), BF16), gk: T.Tensor((HD,), BF16),
                  cos: T.Tensor((T_, ROT), BF16), sin: T.Tensor((T_, ROT), BF16),
                  q: T.Tensor((T_, N_Q, HD), BF16), k: T.Tensor((T_, N_KV, HD), BF16),
                  v: T.Tensor((T_, N_KV, HD), BF16)):
        with T.Kernel(T_, threads=1024) as t:
            tx = T.get_thread_binding()
            lane = tx % 32
            w = tx // 32
            if w < N_Q:
                norm_rope(qkv, cos, sin, t, lane, w * 2 * HD, gq, q, w)
            elif w < N_Q + N_KV:
                norm_rope(qkv, cos, sin, t, lane, K_COL + (w - N_Q) * HD, gk, k, w - N_Q)
            else:
                for i in T.vectorized(8):
                    v[t, w - N_Q - N_KV, 8 * lane + i] = qkv[t, V_COL + (w - N_Q - N_KV) * HD + 8 * lane + i]

    return attn_prep


# ---------------------------------------------------------------------------
# Kernel B: attn_core
# ---------------------------------------------------------------------------
def make_core(T_: int, heads_per_cta: int, gate_mode: str, BN: int = 64, ST: int = 2, DSPLIT: int = 1):
    """Causal GQA flash attention + sigmoid output gate, hand warp-specialized.

    CTA = (``heads_per_cta`` query heads of one KV group) x (64 query tokens) x
    (1/``DSPLIT`` of the 256 output dims); warpgroup 0 is the producer, warpgroup 1 + w
    computes query head ha + w. Grid ``(24 / heads_per_cta * DSPLIT, T / 64)``;
    blockIdx.y walks the token blocks from the last (longest causal range) to the
    first, heads on the fast axis, so the block scheduler starts the longest CTAs first
    (LPT order). With DSPLIT = 2 the two CTAs of a (head, token block) both compute
    S = Q K^T and the softmax, and each does P V and the epilogue for 128 output dims.

    Producer warpgroup (warp 0: Q tiles + K ring, warp 1: V ring, TMA + mbarriers):
      K and V each have an ST-deep ring of BN-row tiles; every consumer warpgroup
      releases a stage (count 128 per warpgroup) once its wgmma has read it.
    Consumer warpgroup, FA3-style intra-warpgroup overlap:
      issue S_{j+1} = Q K_{j+1}^T, rescale O, issue O += P_j V_j, wait S_{j+1},
      softmax(S_{j+1}) while the PV wgmma runs, wait PV, P = bf16(exp2(S)).
      P is converted after the PV wait: converting into a second register buffer while
      the PV wgmma is in flight makes ptxas serialize every wgmma (C7513).
    Gate, y = bf16(bf16(O / l) * bf16(sigmoid(gate))), TMA store:
      gate_mode "early" (heads_per_cta 1): warp 1 TMA-loads the gate tile into its own
          buffer first; producer warps turn it into bf16(sigmoid) in place while the
          consumer runs (all 4 warps when T <= 64 * ST, so their TMA loops never block,
          else warps 2-3); the epilogue only multiplies and stages y in that buffer;
      gate_mode "late": warp 0 loads the gate of head ha + w into the K ring slot
          nkb + w after the last K tile (the stage freed by K tile nkb - 2 + w); the
          consumer warpgroup applies the sigmoid in its epilogue, y staged in Q's tile.

    Rounding (HIR ``attend``): scores f32, running max / sum f32, P landed in bf16 for
    the PV wgmma, f32 accumulation, normalised at the end, landed in bf16. Compiled
    with fast math (ex2.approx / rcp.approx in softmax and sigmoid).
    """
    assert heads_per_cta in (1, 2) and gate_mode in ("early", "late")
    HPC = heads_per_cta
    TM = 64
    D = HD
    NB = T_ // TM
    NT = N_Q // HPC
    EARLY = gate_mode == "early"
    assert DSPLIT == 1 or (EARLY and HPC == 1)
    DV = D // DSPLIT
    assert not EARLY or HPC == 1, "early gate mode needs one head per CTA"
    SIGW = 4 if NB <= ST else 2   # producer warps that compute the gate sigmoid
    c = LOG2E / 16.0          # scale 1/16 and the change to base 2
    NEG = -T.infinity(F32)
    swz = tilelang.layout.make_swizzled_layout

    @T.macro
    def softmax(S, m, mp, mc, l, al, rs, j, bx):
        if (j + 1) * BN - 1 > bx * TM:   # only blocks that cross the diagonal are masked
            for i, cc in T.Parallel(TM, BN):
                S[i, cc] = T.if_then_else(j * BN + cc <= bx * TM + i, S[i, cc], NEG)
        T.copy(m, mp)
        T.reduce_max(S, m, dim=1, clear=False)
        for i in T.Parallel(TM):
            al[i] = T.exp2((mp[i] - m[i]) * c)
            mc[i] = m[i] * c
        for i, cc in T.Parallel(TM, BN):
            S[i, cc] = T.exp2(S[i, cc] * c - mc[i])
        T.reduce_sum(S, rs, dim=1)
        for i in T.Parallel(TM):
            l[i] = l[i] * al[i] + rs[i]

    @T.macro
    def consumer(w, hd, dh, bx, nkb, Qs, Ks, Vs, Gs, y, S, P, O, Ob, G, m, mp, mc, l, al, rs,
                 q_full, k_full, k_empty, v_full, v_empty, g_ready):
        if HPC == 2:
            T.inc_max_nreg(232)
        T.fill(O, 0)
        T.fill(l, 0)
        T.fill(m, NEG)
        T.mbarrier_wait_parity(q_full[0], 0)
        T.mbarrier_wait_parity(k_full[0], 0)
        T.wgmma_gemm(Qs[w, :, :], Ks[0, :, :], S, transpose_B=True, clear_accum=True)
        T.wait_wgmma(0)
        T.mbarrier_arrive(k_empty[0])
        softmax(S, m, mp, mc, l, al, rs, 0, bx)
        T.copy(S, P)
        for j in T.serial(nkb - 1):
            s1 = (j + 1) % ST
            T.mbarrier_wait_parity(k_full[s1], ((j + 1) // ST) % 2)
            T.wgmma_gemm(Qs[w, :, :], Ks[s1, :, :], S, transpose_B=True, clear_accum=True)
            for i, cc in T.Parallel(TM, DV):
                O[i, cc] = O[i, cc] * al[i]
            s0 = j % ST
            T.mbarrier_wait_parity(v_full[s0], (j // ST) % 2)
            T.wgmma_gemm(P, Vs[s0, :, :], O)
            T.wait_wgmma(1)
            T.mbarrier_arrive(k_empty[s1])
            softmax(S, m, mp, mc, l, al, rs, j + 1, bx)
            T.wait_wgmma(0)
            T.mbarrier_arrive(v_empty[s0])
            T.copy(S, P)
        sl = (nkb - 1) % ST
        for i, cc in T.Parallel(TM, DV):
            O[i, cc] = O[i, cc] * al[i]
        T.mbarrier_wait_parity(v_full[sl], ((nkb - 1) // ST) % 2)
        T.wgmma_gemm(P, Vs[sl, :, :], O)
        T.wait_wgmma(0)
        T.mbarrier_arrive(v_empty[sl])
        for i in T.Parallel(TM):
            l[i] = 1.0 / l[i]
        for i, cc in T.Parallel(TM, DV):
            Ob[i, cc] = T.cast(O[i, cc] * l[i], BF16)
        if EARLY:
            T.mbarrier_wait_parity(g_ready[0], 0)
            T.copy(Gs, G)
            for i, cc in T.Parallel(TM, DV):
                Ob[i, cc] = T.cast(T.cast(Ob[i, cc], F32) * T.cast(G[i, cc], F32), BF16)
            T.sync_threads(8 + w, 128)
            T.copy(Ob, Gs)
            T.fence_proxy_async()
            T.sync_threads(8 + w, 128)
            T.copy(Gs, y[bx * TM:(bx + 1) * TM, hd * D + dh * DV:hd * D + (dh + 1) * DV])
        else:
            T.copy(Ob, Qs[w, :, :])
            T.mbarrier_wait_parity(g_ready[w], 0)
            gs = (nkb + w) % ST
            for i, cc in T.Parallel(TM, D):
                Qs[w, i, cc] = T.cast(T.cast(Qs[w, i, cc], F32) * T.cast(T.cast(
                    1.0 / (1.0 + T.exp2(T.cast(Ks[gs, i, cc], F32) * -LOG2E)), BF16), F32), BF16)
            T.copy(Qs[w, :, :], y[bx * TM:(bx + 1) * TM, hd * D:(hd + 1) * D])

    @T.prim_func
    def attn_core(q: T.Tensor((T_, N_Q, D), BF16), k: T.Tensor((T_, N_KV, D), BF16),
                  v: T.Tensor((T_, N_KV, D), BF16), qkv: T.Tensor((T_, QKV_W), BF16),
                  y: T.Tensor((T_, Y_W), BF16)):
        with T.Kernel(NT * DSPLIT, NB, threads=128 * (HPC + 1)) as (hx, by):
            hp = hx // DSPLIT
            dh = hx % DSPLIT
            bx = NB - 1 - by
            ha = hp * HPC
            g = ha // GROUP
            nkb = T.ceildiv((bx + 1) * TM, BN)
            Qs = T.alloc_shared((HPC, TM, D), BF16)
            Ks = T.alloc_shared((ST, BN, D), BF16)
            Vs = T.alloc_shared((ST, BN, DV), BF16)
            if EARLY:
                Gs = T.alloc_shared((TM, DV), BF16)
                G0 = T.alloc_fragment((TM, DV), BF16)
            else:
                Gs = None
                G0 = None
            S0 = T.alloc_fragment((TM, BN), F32)
            P0 = T.alloc_fragment((TM, BN), BF16)
            O0 = T.alloc_fragment((TM, DV), F32)
            Ob0 = T.alloc_fragment((TM, DV), BF16)
            m0 = T.alloc_fragment((TM,), F32)
            mp0 = T.alloc_fragment((TM,), F32)
            mc0 = T.alloc_fragment((TM,), F32)
            l0 = T.alloc_fragment((TM,), F32)
            al0 = T.alloc_fragment((TM,), F32)
            rs0 = T.alloc_fragment((TM,), F32)
            if HPC == 2:
                S1 = T.alloc_fragment((TM, BN), F32)
                P1 = T.alloc_fragment((TM, BN), BF16)
                O1 = T.alloc_fragment((TM, D), F32)
                Ob1 = T.alloc_fragment((TM, D), BF16)
                m1 = T.alloc_fragment((TM,), F32)
                mp1 = T.alloc_fragment((TM,), F32)
                mc1 = T.alloc_fragment((TM,), F32)
                l1 = T.alloc_fragment((TM,), F32)
                al1 = T.alloc_fragment((TM,), F32)
                rs1 = T.alloc_fragment((TM,), F32)
            q_full = T.alloc_barrier(32)
            k_full = T.alloc_barrier([32] * ST)
            k_empty = T.alloc_barrier([128 * HPC] * ST)
            v_full = T.alloc_barrier([32] * ST)
            v_empty = T.alloc_barrier([128 * HPC] * ST)
            g_full = T.alloc_barrier([32] * HPC)
            if EARLY:
                g_ready = T.alloc_barrier(32 * SIGW)
                T.annotate_layout({Qs: swz(Qs), Ks: swz(Ks), Vs: swz(Vs), Gs: swz(Gs)})
            else:
                g_ready = g_full
                T.annotate_layout({Qs: swz(Qs), Ks: swz(Ks), Vs: swz(Vs)})
            tx = T.get_thread_binding()
            if tx < 128:
                with T.ws(0):
                    if HPC == 2:
                        T.dec_max_nreg(40)
                    wid = tx // 32
                    if wid == 0:
                        for w in T.unroll(HPC):
                            T.tma_copy(q[bx * TM:(bx + 1) * TM, ha + w, :], Qs[w, :, :], barrier=q_full[0],
                                       leader_scope_threads=32)
                        T.mbarrier_arrive(q_full[0])
                        for j in T.serial(nkb):
                            s = j % ST
                            if j >= ST:
                                T.mbarrier_wait_parity(k_empty[s], ((j // ST) % 2) ^ 1)
                            T.tma_copy(k[j * BN:(j + 1) * BN, g, :], Ks[s, :, :], barrier=k_full[s],
                                       leader_scope_threads=32)
                            T.mbarrier_arrive(k_full[s])
                        if not EARLY:
                            for w in T.unroll(HPC):
                                # gate of head ha + w -> K ring slot nkb + w
                                T.mbarrier_wait_parity(k_empty[(nkb + w) % ST], (((nkb + w) // ST) % 2) ^ 1)
                                T.tma_copy(qkv[bx * TM:(bx + 1) * TM, (ha + w) * 2 * D + D:(ha + w) * 2 * D + 2 * D],
                                           Ks[(nkb + w) % ST, :, :], barrier=g_full[w], leader_scope_threads=32)
                                T.mbarrier_arrive(g_full[w])
                    elif wid == 1:
                        if EARLY:
                            T.tma_copy(qkv[bx * TM:(bx + 1) * TM, ha * 2 * D + D + dh * DV:ha * 2 * D + D + (dh + 1) * DV], Gs,
                                       barrier=g_full[0], leader_scope_threads=32)
                            T.mbarrier_arrive(g_full[0])
                        for j in T.serial(nkb):
                            s = j % ST
                            if j >= ST:
                                T.mbarrier_wait_parity(v_empty[s], ((j // ST) % 2) ^ 1)
                            T.tma_copy(v[j * BN:(j + 1) * BN, g, dh * DV:(dh + 1) * DV], Vs[s, :, :], barrier=v_full[s],
                                       leader_scope_threads=32)
                            T.mbarrier_arrive(v_full[s])
                    if EARLY and SIGW == 2:
                        if wid >= 2:
                            T.mbarrier_wait_parity(g_full[0], 0)
                            for it in T.unroll(TM * DV // 8 // 64, unroll_factor=8):
                                idx = (tx - 64) + 64 * it
                                for e in T.vectorized(8):
                                    Gs[idx // (DV // 8), (idx % (DV // 8)) * 8 + e] = T.cast(1.0 / (1.0 + T.exp2(
                                        T.cast(Gs[idx // (DV // 8), (idx % (DV // 8)) * 8 + e], F32) * -LOG2E)), BF16)
                            T.mbarrier_arrive(g_ready[0])
                    if EARLY and SIGW == 4:
                        T.mbarrier_wait_parity(g_full[0], 0)
                        for i, cc in T.Parallel(TM, DV):
                            Gs[i, cc] = T.cast(1.0 / (1.0 + T.exp2(T.cast(Gs[i, cc], F32) * -LOG2E)), BF16)
                        T.mbarrier_arrive(g_ready[0])
            elif tx < 256:
                with T.ws(1):
                    consumer(0, ha, dh, bx, nkb, Qs, Ks, Vs, Gs, y, S0, P0, O0, Ob0, G0, m0, mp0, mc0, l0, al0, rs0,
                             q_full, k_full, k_empty, v_full, v_empty, g_ready)
            else:
                if HPC == 2:
                    with T.ws(2):
                        consumer(1, ha + 1, dh, bx, nkb, Qs, Ks, Vs, Gs, y, S1, P1, O1, Ob1, G0, m1, mp1, mc1, l1, al1,
                                 rs1, q_full, k_full, k_empty, v_full, v_empty, g_ready)

    return attn_core


#: Core kernel configuration per prompt length: (query heads per CTA, gate mode, DSPLIT).
#: Measured on GPU0 at 1500 MHz, see ATTENTION_NOTES.md: one head per CTA keeps more
#: CTAs busy at short T and T = 128 also splits the output dims over 2 CTAs (96 CTAs);
#: two heads per CTA share every K/V tile between two consumer warpgroups and win at
#: T = 1024; the "early" gate (sigmoid by producer warps during the main loop) wins up
#: to T = 256.
CORE_CONFIG = {128: (1, "early", 2), 256: (1, "early", 1), 512: (1, "late", 1), 1024: (2, "late", 1)}


def core_pass_configs(gate_mode: str) -> dict:
    """tilelang pass configs of the core kernel.  The early-gate kernel places its own
    shared-memory barriers (fence.proxy.async + bar.sync per consumer warpgroup); the
    automatic insertion would put a bar.sync inside every iteration of the producer's
    in-place sigmoid loop."""
    pc = {"tl.enable_fast_math": True}
    if gate_mode == "early":
        pc["tl.disable_thread_storage_sync"] = True
    return pc


_COMPILED: dict = {}


def _compiled(kind: str, T_: int):
    key = (kind, T_)
    if key not in _COMPILED:
        if kind == "prep":
            _COMPILED[key] = tilelang.compile(make_prep(T_), target="cuda")
        else:
            hpc, gate_mode, dsplit = CORE_CONFIG[T_]
            _COMPILED[key] = tilelang.compile(make_core(T_, hpc, gate_mode, DSPLIT=dsplit), target="cuda",
                                              pass_configs=core_pass_configs(gate_mode))
    return _COMPILED[key]


def _check(name: str, t: torch.Tensor, shape: tuple) -> None:
    if tuple(t.shape) != shape:
        raise ValueError(f"{name}: shape {tuple(t.shape)}, expected {shape}")
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name}: dtype {t.dtype}, expected torch.bfloat16")
    if not t.is_cuda:
        raise ValueError(f"{name}: must be a CUDA tensor")
    if not t.is_contiguous():
        raise ValueError(f"{name}: must be contiguous (strides {t.stride()})")


class AttentionKernels:
    """Compiled prep/core kernels for one prompt length T (128, 256, 512 or 1024).

    ``prep(qkv, gamma_q, gamma_k, cos, sin, q, k, v)``
        q = partial_rope(rms_norm(query, 1 + gamma_q)), k likewise with gamma_k,
        v = copy of the value columns of qkv.  q and k are bitwise equal to the HIR.
    ``core(q, k, v, qkv, y)``
        y = bf16(attend(q, k, v) * bf16(sigmoid(gate))), gate read from qkv
        (gate[t, h, :] = qkv[t, h*512 + 256 : h*512 + 512]).
    All outputs are preallocated by the caller; see the module docstring for shapes.
    Set ``check=False`` to skip the per-call shape/dtype/contiguity checks.
    """

    def __init__(self, T: int, check: bool = True):
        if T not in SUPPORTED_T:
            raise ValueError(f"T={T} not in {SUPPORTED_T}")
        self.T = T
        self.check = check
        self.heads_per_cta, self.gate_mode, self.dsplit = CORE_CONFIG[T]
        self._prep = _compiled("prep", T)
        self._core = _compiled("core", T)

    def prep(self, qkv, gamma_q, gamma_k, cos, sin, q, k, v) -> None:
        if self.check:
            T_ = self.T
            _check("qkv", qkv, (T_, QKV_W))
            _check("gamma_q", gamma_q, (HD,))
            _check("gamma_k", gamma_k, (HD,))
            _check("cos", cos, (T_, ROT))
            _check("sin", sin, (T_, ROT))
            _check("q", q, (T_, N_Q, HD))
            _check("k", k, (T_, N_KV, HD))
            _check("v", v, (T_, N_KV, HD))
        self._prep(qkv, gamma_q, gamma_k, cos, sin, q, k, v)

    def core(self, q, k, v, qkv, y) -> None:
        if self.check:
            T_ = self.T
            _check("q", q, (T_, N_Q, HD))
            _check("k", k, (T_, N_KV, HD))
            _check("v", v, (T_, N_KV, HD))
            _check("qkv", qkv, (T_, QKV_W))
            _check("y", y, (T_, Y_W))
        self._core(q, k, v, qkv, y)

    def kernel_source(self, which: str) -> str:
        """Generated CUDA source of 'prep' or 'core'."""
        return (self._prep if which == "prep" else self._core).get_kernel_source()
