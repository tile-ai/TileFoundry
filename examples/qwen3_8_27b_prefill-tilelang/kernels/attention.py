"""Causal GQA flash attention with the output gate, and the closing norm + lm_head.

attention_tir (runtime, n > 128) tir/attention_{n}.py: one CTA per (query head,
              128 query rows), two consumer warpgroups of 64 rows in FA3
              ping-pong, one TMA producer warpgroup; key blocks of 80 rows; V
              in the TIR's smem arrangement (five 16-key slabs), the value
              product as five RS WGMMAs; mask row / column numbers from the
              iota table; shared buffers at the TIR's offsets.
attention_pp  the version before the TIR alignment (BENCH.md section 10).
attention_ws  (runtime, n = 128) attention_pp without the ping-pong.
              Both read q, k, v and the gate in place from the qkv GEMM's
              output (M, 14336) -- per head [q 256 | gate 256] for 24 heads,
              then 4 k heads, then 4 v heads. Online softmax in f32 (exp2 with
              log2 e folded into the scale), P lands in bf16 for the value
              product, key blocks above the diagonal are skipped. Epilogue, in
              model.py's order: out = bf16(bf16(o / l) * bf16(sigmoid(gate))).
final         the last row's RMSNorm (each CTA recomputes it: 5120 values) and
              a slice of the lm_head GEMV, so one launch produces the logits.
"""

import tilelang
import tilelang.language as T

BF16, F32 = "bfloat16", "float32"
HQ, HKV, D = 24, 4, 256
GQA = HQ // HKV
QKVW = 14336
K0, V0 = 12288, 13312
LOG2E = 1.4426950408889634
# Shared memory as the TIR places it (tools/smem_plan.py): each buffer its own
# region, in name order (sm<k>_...: the TIR's address order).
SMEM_AS_TIR = {"tl.disable_shared_memory_reuse": True}


def attention_ws_kernel(M: int, BN: int = 112, stages: int = 1, heavy_first: bool = True, consumer_regs: int = 240):
    """Warp-specialized form of `attention_kernel`, BM = 128.

    Threads 0..255: two consumer warpgroups, 64 query rows each (the
    FullRow split of every fragment). Threads 256..383: one producer
    warpgroup that streams K and V blocks through a STAGES-deep TMA ring
    (separate full / empty barriers for K and V, so S = Q K^T can start
    before V lands). No block-wide barrier inside the key loop: each
    consumer warpgroup runs its own online softmax and the two drift apart,
    so one warpgroup's exp2 work overlaps the other's WGMMA."""
    BM = 128
    scale = (D ** -0.5) * LOG2E
    nqb = M // BM

    @T.prim_func
    def attention(QKV: T.Tensor((M, QKVW), BF16), Out: T.Tensor((M, HQ * D), BF16)):
        with T.Kernel(nqb, HQ, threads=384) as (by, h):
            bx = (nqb - 1 - by) if heavy_first else by
            g = h // GQA
            Qs = T.alloc_shared((BM, D), BF16)
            Ks = T.alloc_shared((stages, BN, D), BF16)
            Vs = T.alloc_shared((stages, BN, D), BF16)
            s = T.alloc_fragment((BM, BN), F32)
            p = T.alloc_fragment((BM, BN), BF16)
            o = T.alloc_fragment((BM, D), F32)
            m = T.alloc_fragment((BM,), F32)
            m_prev = T.alloc_fragment((BM,), F32)
            corr = T.alloc_fragment((BM,), F32)
            rsum = T.alloc_fragment((BM,), F32)
            l = T.alloc_fragment((BM,), F32)
            qbar = T.alloc_barrier([128])
            kfull = T.alloc_barrier([128] * stages)
            vfull = T.alloc_barrier([128] * stages)
            kempty = T.alloc_barrier([256] * stages)
            vempty = T.alloc_barrier([256] * stages)
            q0 = bx * BM
            nblk = T.ceildiv(q0 + BM, BN)
            if T.get_thread_binding() >= 256:
                T.set_max_nreg(24, 0)
                T.tma_copy(QKV[q0:q0 + BM, h * 2 * D:h * 2 * D + D], Qs, barrier=qbar[0])
                T.mbarrier_arrive(qbar[0])
                for kb in T.serial(nblk):
                    st = kb % stages
                    ph = (kb // stages) % 2
                    k0 = kb * BN
                    T.mbarrier_wait_parity(kempty[st], ph ^ 1)
                    T.tma_copy(QKV[k0:k0 + BN, K0 + g * D:K0 + g * D + D], Ks[st, :, :], barrier=kfull[st])
                    T.mbarrier_arrive(kfull[st])
                    T.mbarrier_wait_parity(vempty[st], ph ^ 1)
                    T.tma_copy(QKV[k0:k0 + BN, V0 + g * D:V0 + g * D + D], Vs[st, :, :], barrier=vfull[st])
                    T.mbarrier_arrive(vfull[st])
            else:
                T.set_max_nreg(consumer_regs, 1)
                T.fill(o, 0)
                T.fill(l, 0)
                T.fill(m, -1e30)
                T.mbarrier_wait_parity(qbar[0], 0)
                for kb in T.serial(nblk):
                    st = kb % stages
                    ph = (kb // stages) % 2
                    k0 = kb * BN
                    T.mbarrier_wait_parity(kfull[st], ph)
                    T.wgmma_gemm(Qs, Ks[st, :, :], s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(kempty[st])
                    if k0 + BN > q0:
                        for i, j in T.Parallel(BM, BN):
                            s[i, j] = T.if_then_else(q0 + i >= k0 + j, s[i, j], -1e30)
                    T.copy(m, m_prev)
                    T.reduce_max(s, m, dim=1, clear=False)
                    for i in T.Parallel(BM):
                        m[i] = T.max(m[i], m_prev[i])
                    for i in T.Parallel(BM):
                        corr[i] = T.exp2(m_prev[i] * scale - m[i] * scale)
                    for i, j in T.Parallel(BM, BN):
                        s[i, j] = T.exp2(s[i, j] * scale - m[i] * scale)
                    T.reduce_sum(s, rsum, dim=1)
                    for i in T.Parallel(BM):
                        l[i] = l[i] * corr[i] + rsum[i]
                    T.copy(s, p)
                    for i, j in T.Parallel(BM, D):
                        o[i, j] *= corr[i]
                    T.mbarrier_wait_parity(vfull[st], ph)
                    T.wgmma_gemm(p, Vs[st, :, :], o, policy=T.GemmWarpPolicy.FullRow)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vempty[st])
                for i, j in T.Parallel(BM, D):
                    ob = T.cast(T.cast(o[i, j] / l[i], BF16), F32)
                    gt = T.cast(QKV[q0 + i, h * 2 * D + D + j], F32)
                    sg = T.cast(T.cast(1.0 / (1.0 + T.exp(-gt)), BF16), F32)
                    Qs[i, j] = T.cast(ob * sg, BF16)
                T.sync_threads(1, 256)
                T.copy(Qs, Out[q0:q0 + BM, h * D:h * D + D])

    return attention


@T.macro
def _softmax_step(s, m, m_prev, corr, rsum, l, q0, k0, BM: int, BN: int, scale: float):
    """Online softmax of one score block in place; corr = rescale factor of o."""
    if k0 + BN > q0:
        for i, j in T.Parallel(BM, BN):
            s[i, j] = T.if_then_else(q0 + i >= k0 + j, s[i, j], -1e30)
    T.copy(m, m_prev)
    T.reduce_max(s, m, dim=1, clear=False)
    for i in T.Parallel(BM):
        m[i] = T.max(m[i], m_prev[i])
    for i in T.Parallel(BM):
        corr[i] = T.exp2(m_prev[i] * scale - m[i] * scale)
    for i, j in T.Parallel(BM, BN):
        s[i, j] = T.exp2(s[i, j] * scale - m[i] * scale)
    T.reduce_sum(s, rsum, dim=1)
    for i in T.Parallel(BM):
        l[i] = l[i] * corr[i] + rsum[i]


def attention_pp_kernel(M: int, BN: int = 80, kst: int = 2, vst: int = 2, heavy_first: bool = True,
                        consumer_regs: int = 240):
    """`attention_ws_kernel` with FA3's ping-pong schedule.

    Iteration j of a consumer warpgroup: [wait its turn] issue S_j = Q K_j^T
    and O += P_{j-1} V_{j-1} (after O *= corr_{j-1}) [hand the turn to the
    other warpgroup], then wait for S_j, run softmax_j, wait for the P V
    product and land P_j in bf16. Two named barriers alternate the turns, so
    one warpgroup's softmax runs under the other's WGMMAs. One score fragment
    and one P fragment, as in the WS kernel: P_{j-1} is read by its WGMMA
    while softmax_j writes the score fragment."""
    BM = 128
    scale = (D ** -0.5) * LOG2E
    nqb = M // BM

    @T.prim_func
    def attention(QKV: T.Tensor((M, QKVW), BF16), Out: T.Tensor((M, HQ * D), BF16)):
        with T.Kernel(nqb, HQ, threads=384) as (by, h):
            bx = (nqb - 1 - by) if heavy_first else by
            g = h // GQA
            Qs = T.alloc_shared((BM, D), BF16)
            Ks = T.alloc_shared((kst, BN, D), BF16)
            Vs = T.alloc_shared((vst, BN, D), BF16)
            s = T.alloc_fragment((BM, BN), F32)
            p = T.alloc_fragment((BM, BN), BF16)
            o = T.alloc_fragment((BM, D), F32)
            m = T.alloc_fragment((BM,), F32)
            m_prev = T.alloc_fragment((BM,), F32)
            corr = T.alloc_fragment((BM,), F32)
            rsum = T.alloc_fragment((BM,), F32)
            l = T.alloc_fragment((BM,), F32)
            qbar = T.alloc_barrier([128])
            kfull = T.alloc_barrier([128] * kst)
            vfull = T.alloc_barrier([128] * vst)
            kempty = T.alloc_barrier([256] * kst)
            vempty = T.alloc_barrier([256] * vst)
            q0 = bx * BM
            nblk = T.ceildiv(q0 + BM, BN)
            tx = T.get_thread_binding()
            if tx >= 256:
                T.set_max_nreg(24, 0)
                T.tma_copy(QKV[q0:q0 + BM, h * 2 * D:h * 2 * D + D], Qs, barrier=qbar[0])
                T.mbarrier_arrive(qbar[0])
                for kb in T.serial(nblk):
                    k0 = kb * BN
                    T.mbarrier_wait_parity(kempty[kb % kst], ((kb // kst) % 2) ^ 1)
                    T.tma_copy(QKV[k0:k0 + BN, K0 + g * D:K0 + g * D + D], Ks[kb % kst, :, :], barrier=kfull[kb % kst])
                    T.mbarrier_arrive(kfull[kb % kst])
                    T.mbarrier_wait_parity(vempty[kb % vst], ((kb // vst) % 2) ^ 1)
                    T.tma_copy(QKV[k0:k0 + BN, V0 + g * D:V0 + g * D + D], Vs[kb % vst, :, :], barrier=vfull[kb % vst])
                    T.mbarrier_arrive(vfull[kb % vst])
            else:
                T.set_max_nreg(consumer_regs, 1)
                T.fill(o, 0)
                T.fill(l, 0)
                T.fill(m, -1e30)
                T.mbarrier_wait_parity(qbar[0], 0)
                # warpgroup 1 hands warpgroup 0 the first turn
                if tx >= 128:
                    T.named_barrier_arrive(8, 256)
                # block 0: S_0 and its softmax, no P V yet
                if tx < 128:
                    T.sync_threads(8, 256)
                else:
                    T.sync_threads(9, 256)
                T.mbarrier_wait_parity(kfull[0], 0)
                T.wgmma_gemm(Qs, Ks[0, :, :], s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                if tx < 128:
                    T.named_barrier_arrive(9, 256)
                else:
                    T.named_barrier_arrive(8, 256)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kempty[0])
                _softmax_step(s, m, m_prev, corr, rsum, l, q0, 0, BM, BN, scale)
                T.copy(s, p)
                for kb in T.serial(1, nblk):
                    if tx < 128:
                        T.sync_threads(8, 256)
                    else:
                        T.sync_threads(9, 256)
                    T.mbarrier_wait_parity(kfull[kb % kst], (kb // kst) % 2)
                    T.wgmma_gemm(Qs, Ks[kb % kst, :, :], s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                    for i, j in T.Parallel(BM, D):
                        o[i, j] *= corr[i]
                    T.mbarrier_wait_parity(vfull[(kb - 1) % vst], ((kb - 1) // vst) % 2)
                    T.wgmma_gemm(p, Vs[(kb - 1) % vst, :, :], o, policy=T.GemmWarpPolicy.FullRow)
                    if tx < 128:
                        T.named_barrier_arrive(9, 256)
                    else:
                        T.named_barrier_arrive(8, 256)
                    T.wait_wgmma(1)
                    T.mbarrier_arrive(kempty[kb % kst])
                    _softmax_step(s, m, m_prev, corr, rsum, l, q0, kb * BN, BM, BN, scale)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vempty[(kb - 1) % vst])
                    T.copy(s, p)
                for i, j in T.Parallel(BM, D):
                    o[i, j] *= corr[i]
                T.mbarrier_wait_parity(vfull[(nblk - 1) % vst], ((nblk - 1) // vst) % 2)
                T.wgmma_gemm(p, Vs[(nblk - 1) % vst, :, :], o, policy=T.GemmWarpPolicy.FullRow)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vempty[(nblk - 1) % vst])
                for i, j in T.Parallel(BM, D):
                    ob = T.cast(T.cast(o[i, j] / l[i], BF16), F32)
                    gt = T.cast(QKV[q0 + i, h * 2 * D + D + j], F32)
                    sg = T.cast(T.cast(1.0 / (1.0 + T.exp(-gt)), BF16), F32)
                    Qs[i, j] = T.cast(ob * sg, BF16)
                T.sync_threads(1, 256)
                T.copy(Qs, Out[q0:q0 + BM, h * D:h * D + D])

    return attention


@T.macro
def _softmax_tir(s, m, m_prev, corr, rsum, l, rows, cols, Iota, q0, k0, BM: int, BN: int, scale: float):
    """`_softmax_step` with the mask of tir/attention_{n}.py: on a block the
    diagonal crosses, cols = iota[0:BN] + k0 is read into registers (one per
    score-fragment column the thread holds), dif = rows - cols (rows = iota +
    q0, held per score-fragment row), s = where(dif >= 0)."""
    if k0 + BN > q0:
        for j in T.Parallel(BN):
            cols[j] = Iota[j] + k0
        for i, j in T.Parallel(BM, BN):
            s[i, j] = T.if_then_else(rows[i] - cols[j] >= 0, s[i, j], -1e30)
    T.copy(m, m_prev)
    T.reduce_max(s, m, dim=1, clear=False)
    for i in T.Parallel(BM):
        m[i] = T.max(m[i], m_prev[i])
    for i in T.Parallel(BM):
        corr[i] = T.exp2(m_prev[i] * scale - m[i] * scale)
    for i, j in T.Parallel(BM, BN):
        s[i, j] = T.exp2(s[i, j] * scale - m[i] * scale)
    T.reduce_sum(s, rsum, dim=1)
    for i in T.Parallel(BM):
        l[i] = l[i] * corr[i] + rsum[i]


@T.macro
def _land_p(s, p0, p1, p2, p3, p4, BM: int):
    """P_kb in bf16, one (BM, 16) fragment per 16-key slab (the RS WGMMA's A)."""
    for i, j in T.Parallel(BM, 16):
        p0[i, j] = T.cast(s[i, j], BF16)
    for i, j in T.Parallel(BM, 16):
        p1[i, j] = T.cast(s[i, 16 + j], BF16)
    for i, j in T.Parallel(BM, 16):
        p2[i, j] = T.cast(s[i, 32 + j], BF16)
    for i, j in T.Parallel(BM, 16):
        p3[i, j] = T.cast(s[i, 48 + j], BF16)
    for i, j in T.Parallel(BM, 16):
        p4[i, j] = T.cast(s[i, 64 + j], BF16)


@T.macro
def _pv(p0, p1, p2, p3, p4, Vs, st, o):
    """O += P V: one m64n256k16 RS WGMMA per 16-key slab of V."""
    T.wgmma_gemm(p0, Vs[st, 0, :, :], o, policy=T.GemmWarpPolicy.FullRow)
    T.wgmma_gemm(p1, Vs[st, 1, :, :], o, policy=T.GemmWarpPolicy.FullRow)
    T.wgmma_gemm(p2, Vs[st, 2, :, :], o, policy=T.GemmWarpPolicy.FullRow)
    T.wgmma_gemm(p3, Vs[st, 3, :, :], o, policy=T.GemmWarpPolicy.FullRow)
    T.wgmma_gemm(p4, Vs[st, 4, :, :], o, policy=T.GemmWarpPolicy.FullRow)


def attention_tir_kernel(M: int, BN: int = 80, kst: int = 2, vst: int = 2, consumer_regs: int = 240, IOTA: str = "int32"):
    """tir/attention_{M}.py (scheduled/attention.py): the TIR's per-block order.

    Block kb of a consumer warpgroup: S_kb = Q K_kb^T; softmax_kb (mask on a
    block the diagonal crosses, row and column numbers from the iota table);
    O *= corr_kb; O += P_kb V_kb -- issued as the TIR orders them (the loop is
    rotated: an iteration issues block kb - 1's value product, then block kb's
    scores, and waits for both). V lies in smem as the TIR lays it, five 16-key
    slabs of [4 x 64 columns] (16 x 256 each, SW128), loaded by five TMA
    copies; the value product is five m64n256k16 RS WGMMAs, one per slab.
    Named barriers alternate the two warpgroups' turns at the tensor cores
    (FA3's ping-pong between warpgroups: the backend's synchronization), so
    one warpgroup's softmax runs under the other's WGMMAs. Issuing S_kb ahead
    of block kb - 1's value product (attention_pp's intra-warpgroup overlap)
    measured 1647.5 against 1657.0 us at 8192, 107.6 / 108.0 at 1024."""
    BM = 128
    scale = (D ** -0.5) * LOG2E
    nqb = M // BM
    NS = BN // 16
    assert NS == 5

    @T.prim_func
    def attention(QKV: T.Tensor((M, QKVW), BF16), Iota: T.Tensor((BM,), IOTA), Out: T.Tensor((M, HQ * D), BF16)):
        with T.Kernel(nqb, HQ, threads=384) as (by, h):
            bx = nqb - 1 - by
            g = h // GQA
            sm0_qs = T.alloc_shared((BM, D), BF16)
            sm1_ks = T.alloc_shared((kst, BN, D), BF16)
            sm2_vs = T.alloc_shared((vst, NS, 16, D), BF16)
            s = T.alloc_fragment((BM, BN), F32)
            p0 = T.alloc_fragment((BM, 16), BF16)
            p1 = T.alloc_fragment((BM, 16), BF16)
            p2 = T.alloc_fragment((BM, 16), BF16)
            p3 = T.alloc_fragment((BM, 16), BF16)
            p4 = T.alloc_fragment((BM, 16), BF16)
            o = T.alloc_fragment((BM, D), F32)
            m = T.alloc_fragment((BM,), F32)
            m_prev = T.alloc_fragment((BM,), F32)
            corr = T.alloc_fragment((BM,), F32)
            rsum = T.alloc_fragment((BM,), F32)
            l = T.alloc_fragment((BM,), F32)
            rows = T.alloc_fragment((BM,), IOTA)
            cols = T.alloc_fragment((BN,), IOTA)
            qbar = T.alloc_barrier([128])
            kfull = T.alloc_barrier([128] * kst)
            vfull = T.alloc_barrier([128] * vst)
            kempty = T.alloc_barrier([256] * kst)
            vempty = T.alloc_barrier([256] * vst)
            q0 = bx * BM
            nblk = T.ceildiv(q0 + BM, BN)
            tx = T.get_thread_binding()
            if tx >= 256:
                T.set_max_nreg(24, 0)
                T.tma_copy(QKV[q0:q0 + BM, h * 2 * D:h * 2 * D + D], sm0_qs, barrier=qbar[0])
                T.mbarrier_arrive(qbar[0])
                for kb in T.serial(nblk):
                    k0 = kb * BN
                    T.mbarrier_wait_parity(kempty[kb % kst], ((kb // kst) % 2) ^ 1)
                    T.tma_copy(QKV[k0:k0 + BN, K0 + g * D:K0 + g * D + D], sm1_ks[kb % kst, :, :], barrier=kfull[kb % kst])
                    T.mbarrier_arrive(kfull[kb % kst])
                    T.mbarrier_wait_parity(vempty[kb % vst], ((kb // vst) % 2) ^ 1)
                    for r in T.unroll(NS):
                        T.tma_copy(QKV[k0 + r * 16:k0 + r * 16 + 16, V0 + g * D:V0 + g * D + D], sm2_vs[kb % vst, r, :, :], barrier=vfull[kb % vst])
                    T.mbarrier_arrive(vfull[kb % vst])
            else:
                T.set_max_nreg(consumer_regs, 1)
                T.fill(o, 0)
                T.fill(l, 0)
                T.fill(m, -1e30)
                for i in T.Parallel(BM):
                    rows[i] = Iota[i] + q0
                T.mbarrier_wait_parity(qbar[0], 0)
                # warpgroup 1 hands warpgroup 0 the first turn
                if tx >= 128:
                    T.named_barrier_arrive(8, 256)
                # kb = 0: S_0 and its softmax, no P V yet
                if tx < 128:
                    T.sync_threads(8, 256)
                else:
                    T.sync_threads(9, 256)
                T.mbarrier_wait_parity(kfull[0], 0)
                T.wgmma_gemm(sm0_qs, sm1_ks[0, :, :], s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                if tx < 128:
                    T.named_barrier_arrive(9, 256)
                else:
                    T.named_barrier_arrive(8, 256)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kempty[0])
                _softmax_tir(s, m, m_prev, corr, rsum, l, rows, cols, Iota, q0, 0, BM, BN, scale)
                _land_p(s, p0, p1, p2, p3, p4, BM)
                for kb in T.serial(1, nblk):
                    if tx < 128:
                        T.sync_threads(8, 256)
                    else:
                        T.sync_threads(9, 256)
                    # block kb - 1's last statements, then block kb's first (the TIR's order)
                    for i, j in T.Parallel(BM, D):
                        o[i, j] *= corr[i]
                    T.mbarrier_wait_parity(vfull[(kb - 1) % vst], ((kb - 1) // vst) % 2)
                    _pv(p0, p1, p2, p3, p4, sm2_vs, (kb - 1) % vst, o)
                    T.mbarrier_wait_parity(kfull[kb % kst], (kb // kst) % 2)
                    T.wgmma_gemm(sm0_qs, sm1_ks[kb % kst, :, :], s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                    if tx < 128:
                        T.named_barrier_arrive(9, 256)
                    else:
                        T.named_barrier_arrive(8, 256)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vempty[(kb - 1) % vst])
                    T.mbarrier_arrive(kempty[kb % kst])
                    _softmax_tir(s, m, m_prev, corr, rsum, l, rows, cols, Iota, q0, kb * BN, BM, BN, scale)
                    _land_p(s, p0, p1, p2, p3, p4, BM)
                for i, j in T.Parallel(BM, D):
                    o[i, j] *= corr[i]
                T.mbarrier_wait_parity(vfull[(nblk - 1) % vst], ((nblk - 1) // vst) % 2)
                _pv(p0, p1, p2, p3, p4, sm2_vs, (nblk - 1) % vst, o)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vempty[(nblk - 1) % vst])
                for i, j in T.Parallel(BM, D):
                    ob = T.cast(T.cast(o[i, j] / l[i], BF16), F32)
                    gt = T.cast(QKV[q0 + i, h * 2 * D + D + j], F32)
                    sg = T.cast(T.cast(1.0 / (T.exp(-gt) + 1.0), BF16), F32)
                    sm0_qs[i, j] = T.cast(ob * sg, BF16)
                T.sync_threads(1, 256)
                T.copy(sm0_qs, Out[q0:q0 + BM, h * D:h * D + D])

    return attention


def final_kernel(M: int, H: int = 5120, V: int = 248320, RW: int = 4, eps: float = 1e-6):
    """tir/final_{M}.py (scheduled/embed_final.py): logits = lm_head(rmsnorm(hidden[last])).

    V / 32 CTAs x 256 threads. Each CTA first normalises row `last` (thread t
    holds the 8-byte vectors at columns (k * 256 + t) * 4, as rmsnorm_kernel;
    the row is in L2 after the first CTA) into shared memory as bf16; then
    warp w takes vocabulary rows 32 b + 4 w + q: lane l reads 16-byte vectors
    l, l + 32, ... of a row, so the 2.5 GB weight streams in coalesced
    512-byte warp loads with RW * 20 of them in flight per lane."""
    NVEC = H // 8            # 16-byte vectors per row
    PER = NVEC // 32         # per lane
    RB = 8 * RW              # rows per CTA
    TV, NV4 = 256, 4
    KV = H // (TV * NV4)

    @T.prim_func
    def final(Hid: T.Tensor((M, H), BF16), Last: T.Tensor((1,), "int64"), G: T.Tensor((H,), BF16),
              Wh: T.Tensor((V, H), BF16), Logits: T.Tensor((1, V), BF16)):
        with T.Kernel(V // RB, threads=256) as bx:
            sm0_xs = T.alloc_shared((H,), BF16)
            sm1_ws = T.alloc_shared((8,), F32)
            xr = T.alloc_local((KV * NV4,), BF16)
            gr = T.alloc_local((KV * NV4,), BF16)
            nb = T.alloc_local((NV4,), BF16)
            red = T.alloc_local((1,), F32)
            wv = T.alloc_local((8,), BF16)
            xv = T.alloc_local((8,), BF16)
            acc = T.alloc_local((RW,), F32)
            t = T.get_thread_binding()
            r = Last[0]
            for k in T.unroll(KV):
                for j in T.vectorized(NV4):
                    xr[k * NV4 + j] = Hid[r, (k * TV + t) * NV4 + j]
            red[0] = 0.0
            for i in T.unroll(KV * NV4):
                red[0] = red[0] + T.cast(xr[i], F32) * T.cast(xr[i], F32)
            for o in T.unroll(5):
                red[0] = red[0] + T.shfl_xor(red[0], 16 >> o, 32)
            if t % 32 == 0:
                sm1_ws[t // 32] = red[0]
            for k in T.unroll(KV):
                for j in T.vectorized(NV4):
                    gr[k * NV4 + j] = G[(k * TV + t) * NV4 + j]
            T.sync_threads()
            red[0] = 0.0
            for w in T.unroll(8):
                red[0] = red[0] + sm1_ws[w]
            rs = T.rsqrt(red[0] / H + eps)
            for k in T.unroll(KV):
                for j in T.unroll(NV4):
                    nb[j] = T.cast(T.cast(xr[k * NV4 + j], F32) * rs * (T.cast(gr[k * NV4 + j], F32) + 1.0), BF16)
                for j in T.vectorized(NV4):
                    sm0_xs[(k * TV + t) * NV4 + j] = nb[j]
            T.sync_threads()
            tx = T.get_thread_binding()
            lane = tx % 32
            row0 = bx * RB + (tx // 32) * RW
            for q in T.unroll(RW):
                acc[q] = 0.0
            for p in T.unroll(PER):
                for j in T.vectorized(8):
                    xv[j] = sm0_xs[(p * 32 + lane) * 8 + j]
                for q in T.unroll(RW):
                    for j in T.vectorized(8):
                        wv[j] = Wh[row0 + q, (p * 32 + lane) * 8 + j]
                    for j in T.unroll(8):
                        acc[q] = acc[q] + T.cast(wv[j], F32) * T.cast(xv[j], F32)
            for q in T.unroll(RW):
                for o in T.unroll(5):
                    acc[q] = acc[q] + T.shfl_xor(acc[q], 16 >> o, 32)
            if lane == 0:
                for q in T.unroll(RW):
                    Logits[0, row0 + q] = T.cast(acc[q], BF16)

    return final


_CACHE = {}


def compiled(name, *args, **kw):
    key = (name, args, tuple(sorted(kw.items())))
    if key not in _CACHE:
        fn = {"attention_ws": attention_ws_kernel, "attention_pp": attention_pp_kernel, "attention_tir": attention_tir_kernel,
              "final": final_kernel}[name]
        cfg = SMEM_AS_TIR if name in ("attention_tir", "final") else None
        _CACHE[key] = tilelang.compile(fn(*args, **kw), out_idx=None, pass_configs=cfg)
    return _CACHE[key]
