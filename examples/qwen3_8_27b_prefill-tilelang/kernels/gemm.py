"""Persistent warp-specialized GEMM for every prefill projection: C = A @ W^T.

A (M, K) row-major activations, W (N, K) row-major weights -- the checkpoint's
own (out, in) order, so both operands are K-major. One CTA per SM.

Work, in order, for CTA b of 132 (evidence/STEP3.md rows 12-17, STEP4.md):
  1. split tail: the R = T mod 132 tiles that would leave a last wave mostly
     idle are cut along K into s = 132 // R pieces (s <= 8); CTA b < R * s
     takes piece b first and writes its f32 partial to WS[b].
  2. whole tiles: the first T - R tiles in a grouped order (GROUP M-tiles per
     band of N), tile b + 132 w in wave w, so a wave's CTAs read a block of
     tiles whose weight columns stay in L2 (STEP3.md, "where analyze says
     memory").
  3. a second kernel (`gemm_tail_kernel`) sums each tail tile's s partials in
     piece order and runs the epilogue. The handover is a kernel boundary
     because HIR states no reduction across CTAs inside one launch
     (scheduled/gemm.py: the `*_tail` funcs; STEP4.md row 8).

Threads: two consumer warpgroups (0..255) share one BM x BN tile, BM/2 rows
each, one m64nBNk16 WGMMA per 16 of K; one producer warpgroup (256..383)
streams BK-wide K slabs through a STAGES-deep TMA ring guarded by full/empty
mbarriers. A consumer keeps one WGMMA batch in flight and releases a stage
only after the batch that read it retired. Producer and consumers are the
two arms of one branch: two sequential T.ws regions get a __syncthreads()
between them that the producer never reaches (TILELANG_NOTES, reproduced on
0.1.15).

Epilogues -- each lands exactly where model.py's function lands:

  plain        C = bf16(acc)                                    in_proj
  residual     C = bf16(R + bf16(acc))                          o_proj, out_proj, down
  swiglu       W rows interleaved [gate BN/2 | up BN/2] per tile;
               C[:, j] = bf16(bf16(silu(bf16(acc_g))) * bf16(acc_u))   gate_up
  qk_norm_rope q|gate per head, then k, then v (N = 14336); q and k
               heads are RMS-normed with (1 + gamma) and their first 64
               columns rotated, rotate-half, each step landing in bf16   qkv
"""

import tilelang
import tilelang.language as T

NUM_SM = 132
BF16, F32 = "bfloat16", "float32"


def _tile_coords(tile, mb, nb, group):
    """Grouped raster: GROUP M-tiles walk down one N band before moving on."""
    gsz = group * nb
    g = tile // gsz
    first = g * group
    gm = T.min(mb - first, group)
    r = tile % gsz
    return first + r % gm, r // gm


# Experiment override for the segmentation study (tools/segment_study.py):
# {"split_tail": bool or None, "group": int or None}. None = the length rule.
OVERRIDE = {"split_tail": None, "group": None}


def default_group(M, BM=128):
    """M-tiles per raster group: 16 where there are that many (4096, 8192), else 8.
    16 keeps a wave's tiles (8 bands x 16 M-tiles, ~43 MB) inside L2 and halves
    the weight's DRAM re-reads; under the 700 W cap that is also clock
    (evidence/STEP4.md row 15: 8 -> 16 -0.6..0.8 %, 32 slower)."""
    mb = M // BM
    if OVERRIDE["group"] and mb % OVERRIDE["group"] == 0:
        return OVERRIDE["group"]
    if mb % 16 == 0:
        return 16
    return 8 if mb % 8 == 0 else mb   # short prompts: one group of all M-tiles


def plan(M, N, K, epilogue="plain", BM=128, BN=256, BK=64, split_tail=True, group=None):
    """The tile / split decomposition `gemm_kernel` uses, as plain numbers.

    The tail is the last R tiles in raster order, R = T mod 132 rounded up to
    whole bands (GROUP tiles), so tail tile (band, mi) needs no division --
    the form scheduled/gemm.py states. Each is cut into s pieces, s the
    largest divisor of the K block count with R * s <= 132 and s <= 8; no
    split when s < 2."""
    group = group or default_group(M, BM)
    if OVERRIDE["split_tail"] is not None:
        split_tail = OVERRIDE["split_tail"]
    mb = (M + BM - 1) // BM
    nb = (N + BN - 1) // BN
    kb = (K + BK - 1) // BK
    tiles = mb * nb
    rem = -(-(tiles % NUM_SM) // group) * group
    splits = 1
    if split_tail and rem:
        splits = max([d for d in range(1, min(NUM_SM // rem, 8) + 1) if kb % d == 0])
    if splits < 2 or rem >= tiles:   # no split without whole tiles (short prompts)
        rem, splits = 0, 1
    whole = tiles - rem
    cy = 1  # tail kernel: row chunks per tile, as many as fit 132 CTAs
    while rem and rem * cy * 2 <= NUM_SM and cy < 8:
        cy *= 2
    return dict(mb=mb, nb=nb, kb=kb, tiles=tiles, rem=rem, splits=splits, whole=whole,
                whole_waves=(whole + NUM_SM - 1) // NUM_SM, units=rem * splits,
                tail_bands=rem // group, band0=nb - rem // group, tail_cy=cy, group=group)


def gemm_kernel(M: int, N: int, K: int, epilogue: str = "plain", BM: int = 128, BN: int = 256,
                BK: int = 64, stages: int = 3, group: int = None, split_tail: bool = True, eps: float = 1e-6):
    p = plan(M, N, K, epilogue, BM, BN, BK, split_tail, group)
    group = p["group"]
    mb, nb, kb = p["mb"], p["nb"], p["kb"]
    rem, splits, whole, whole_waves, units = p["rem"], p["splits"], p["whole"], p["whole_waves"], p["units"]
    kpp = (kb + splits - 1) // splits      # K blocks per split piece
    swiglu = epilogue == "swiglu"
    NO = N // 2 if swiglu else N           # output columns
    BO = BN // 2 if swiglu else BN         # output columns per tile
    HD, ROT, HALF = 256, 64, 32            # qk_norm_rope: head dim, rotary dims
    NQG, NKH = 48, 4                       # q|gate column blocks, k heads
    WSU = max(units, 1)                    # workspace pieces

    def _uses(n, s):
        """How many of K blocks 0 .. n-1 fall in stage s."""
        return (n + stages - 1 - s) // stages

    @T.macro
    def producer(A, W, As, Ws, full, empty, bid):
        T.set_max_nreg(40, 0)
        if units > 0:  # Python-level: no dead split code (its tile math folds to a 0 divisor)
            if bid < units:
                tile = whole + bid // splits
                bm, bn = _tile_coords(tile, mb, nb, group)
                k0 = (bid % splits) * kpp
                for k in T.serial(kpp):
                    if k0 + k < kb:
                        s = k % stages
                        T.mbarrier_wait_parity(empty[s], ((k // stages) % 2) ^ 1)
                        T.tma_copy(A[bm * BM:(bm + 1) * BM, (k0 + k) * BK:(k0 + k + 1) * BK], As[s, :, :], barrier=full[s])
                        T.tma_copy(W[bn * BN:(bn + 1) * BN, (k0 + k) * BK:(k0 + k + 1) * BK], Ws[s, :, :], barrier=full[s])
                        T.mbarrier_arrive(full[s])
        base = T.if_then_else(bid < units, T.min(kpp, kb - (bid % splits) * kpp), 0)
        for w in T.serial(whole_waves):
            tile = w * NUM_SM + bid
            if tile < whole:
                bm, bn = _tile_coords(tile, mb, nb, group)
                for k in T.serial(kb):
                    # the TIR's ring: K block k of every tile in stage k % 3;
                    # the phase counts that stage's earlier uses on this CTA
                    s = k % stages
                    u = _uses(base, s) + w * _uses(kb, s) + k // stages
                    T.mbarrier_wait_parity(empty[s], (u % 2) ^ 1)
                    T.tma_copy(A[bm * BM:(bm + 1) * BM, k * BK:(k + 1) * BK], As[s, :, :], barrier=full[s])
                    T.tma_copy(W[bn * BN:(bn + 1) * BN, k * BK:(k + 1) * BK], Ws[s, :, :], barrier=full[s])
                    T.mbarrier_arrive(full[s])

    @T.macro
    def mainloop(As, Ws, acc, full, empty, pbase, wt, nk):
        """nk K blocks of one tile, block k in stage k % 3 (the TIR's ring);
        before it this CTA ran a piece of pbase blocks and wt whole tiles."""
        T.clear(acc)
        for k in T.serial(nk):
            s = k % stages
            u = _uses(pbase, s) + wt * _uses(kb, s) + k // stages
            T.mbarrier_wait_parity(full[s], u % 2)
            T.wgmma_gemm(As[s, :, :], Ws[s, :, :], acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.wait_wgmma(1)
            if k > 0:
                T.mbarrier_arrive(empty[(k + stages - 1) % stages])
        T.wait_wgmma(0)
        T.mbarrier_arrive(empty[(nk - 1) % stages])

    @T.macro
    def consumer(As, Ws, acc, full, empty, bid, WS, epilogue_body, b1, b2, b3, b4, b5, b6, b7):
        T.set_max_nreg(232, 1)
        tx = T.get_thread_binding()
        if units > 0:  # see producer
            if bid < units:
                tile = whole + bid // splits
                bm, bn = _tile_coords(tile, mb, nb, group)
                nk = T.min(kpp, kb - (bid % splits) * kpp)
                mainloop(As, Ws, acc, full, empty, 0, 0, nk)
                T.copy(acc, WS[bid, :, :])  # summed and finished by gemm_tail_kernel
        base = T.if_then_else(bid < units, T.min(kpp, kb - (bid % splits) * kpp), 0)
        for w in T.serial(whole_waves):
            tile = w * NUM_SM + bid
            if tile < whole:
                bm, bn = _tile_coords(tile, mb, nb, group)
                mainloop(As, Ws, acc, full, empty, base, w, kb)
                epilogue_body(acc, bm, bn, b1, b2, b3, b4, b5, b6, b7)

    @T.macro
    def epi_plain(acc, bm, bn, Cs, C, b3, b4, b5, b6, b7):
        T.copy(acc, Cs)
        T.copy(Cs, C[bm * BM, bn * BO])

    @T.macro
    def epi_residual(acc, bm, bn, Cs, R, C, b4, b5, b6, b7):
        T.copy(R[bm * BM, bn * BN], Cs)
        for i, j in T.Parallel(BM, BN):
            Cs[i, j] = T.cast(T.cast(Cs[i, j], F32) + T.cast(T.cast(acc[i, j], BF16), F32), BF16)
        T.copy(Cs, C[bm * BM, bn * BN])

    @T.macro
    def epi_swiglu(acc, bm, bn, Cs, C, b3, b4, b5, b6, b7):
        # tir/gate_up_whole_*.py: cs = bf16(acc), all 256 columns, in SW128
        # smem; on a 16 x 16 grid of the consumer threads, thread (r, c) reads
        # gate (cols c*8 ..) and up (cols 128 + c*8 ..) of rows r*8 .. r*8+7
        # and writes bf16(silu(gate)) * up over its own gate values -- the
        # TIR's `staged`, which shares cs's first half. Then the TMA store.
        T.copy(acc, Cs)
        tx = T.get_thread_binding()
        er = tx // 16
        ec = tx % 16
        gv = T.alloc_local((8,), BF16)
        uv = T.alloc_local((8,), BF16)
        for i in T.unroll(8):
            for j in T.vectorized(8):
                gv[j] = Cs[er * 8 + i, ec * 8 + j]
            for j in T.vectorized(8):
                uv[j] = Cs[er * 8 + i, BO + ec * 8 + j]
            for j in T.unroll(8):
                g = T.cast(gv[j], F32)
                act = T.cast(T.call_extern(F32, "__fdividef", g, 1.0 + T.call_extern(F32, "__expf", -g)), BF16)
                gv[j] = T.cast(T.cast(act, F32) * T.cast(uv[j], F32), BF16)
            for j in T.vectorized(8):
                Cs[er * 8 + i, ec * 8 + j] = gv[j]
        T.copy(Cs[:, 0:BO], C[bm * BM, bn * BO])

    @T.macro
    def qk_norm_rope(bm, Cs, g, cos, sin):
        # one head of tir/qkv_whole_*.py's guarded epilogue: on a 16 x 16 grid
        # of the consumers, thread (er, ec) owns rows er*8 .. er*8+7, columns
        # ec*16 .. ec*16+15: Qwen3_5RMSNorm in f32 (row sum over the 16 lanes
        # of the row), landing in bf16 over cs; then rotate-half on the
        # leading 64 columns, thread (er, ec) taking columns 2 ec, 2 ec + 1 and
        # their partners 32 + 2 ec, 32 + 2 ec + 1; each product and the sum
        # land in bf16 (HF order).
        tx = T.get_thread_binding()
        er = tx // 16
        ec = tx % 16
        xv = T.alloc_local((16,), BF16)
        gw = T.alloc_local((16,), BF16)
        red = T.alloc_local((1,), F32)
        lo = T.alloc_local((2,), F32)
        hi = T.alloc_local((2,), F32)
        for j in T.vectorized(16):
            gw[j] = g[ec * 16 + j]
        T.sync_threads(1, 256)
        for i in T.unroll(8):
            for j in T.vectorized(16):
                xv[j] = Cs[er * 8 + i, ec * 16 + j]
            red[0] = 0.0
            for j in T.unroll(16):
                red[0] = red[0] + T.cast(xv[j], F32) * T.cast(xv[j], F32)
            for o in T.unroll(4):
                red[0] = red[0] + T.shfl_xor(red[0], 8 >> o, 16)
            rs = T.rsqrt(red[0] / HD + eps)
            for j in T.unroll(16):
                xv[j] = T.cast(T.cast(xv[j], F32) * rs * (T.cast(gw[j], F32) + 1.0), BF16)
            for j in T.vectorized(16):
                Cs[er * 8 + i, ec * 16 + j] = xv[j]
        T.sync_threads(1, 256)
        for i in T.unroll(8):
            r = er * 8 + i
            for j in T.unroll(2):
                lo[j] = T.cast(Cs[r, ec * 2 + j], F32)
                hi[j] = T.cast(Cs[r, HALF + ec * 2 + j], F32)
            for j in T.unroll(2):
                c0 = ec * 2 + j
                Cs[r, c0] = T.cast(T.cast(T.cast(lo[j] * T.cast(cos[bm * BM + r, c0], F32), BF16), F32)
                                   + T.cast(T.cast(-hi[j] * T.cast(sin[bm * BM + r, c0], F32), BF16), F32), BF16)
                Cs[r, HALF + c0] = T.cast(T.cast(T.cast(hi[j] * T.cast(cos[bm * BM + r, HALF + c0], F32), BF16), F32)
                                          + T.cast(T.cast(lo[j] * T.cast(sin[bm * BM + r, HALF + c0], F32), BF16), F32), BF16)

    @T.macro
    def epi_qk(acc, bm, bn, Cs, ss, gq, gk, cos, sin, C):
        # tir/qkv_whole_*.py: cs = bf16(acc) staged; the q-head guard (bn even
        # below NQG) with gamma_q, the k-head guard (NQG <= bn < NQG + NKH)
        # with gamma_k; then the TMA store (gate and v tiles as they land).
        T.copy(acc, Cs)
        if bn < NQG and bn % 2 == 0:
            qk_norm_rope(bm, Cs, gq, cos, sin)
        if bn >= NQG and bn < NQG + NKH:
            qk_norm_rope(bm, Cs, gk, cos, sin)
        T.copy(Cs, C[bm * BM, bn * BN])

    if epilogue == "plain":
        @T.prim_func
        def gemm_plain(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16), C: T.Tensor((M, NO), BF16),
                 WS: T.Tensor((WSU, BM, BN), F32)):
            with T.Kernel(NUM_SM, threads=384) as bid:
                sm1_as = T.alloc_shared((stages, BM, BK), BF16)
                sm0_ws = T.alloc_shared((stages, BN, BK), BF16)
                sm2_cs = T.alloc_shared((BM, BO), BF16)
                acc = T.alloc_fragment((BM, BN), F32)
                full = T.alloc_barrier([128] * stages)
                empty = T.alloc_barrier([256] * stages)

                if T.get_thread_binding() >= 256:
                    producer(A, W, sm1_as, sm0_ws, full, empty, bid)
                else:
                    consumer(sm1_as, sm0_ws, acc, full, empty, bid, WS, epi_plain, sm2_cs, C, sm2_cs, sm2_cs, sm2_cs, sm2_cs, sm2_cs)

    elif epilogue == "residual":
        @T.prim_func
        def gemm_residual(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16), R: T.Tensor((M, N), BF16),
                 C: T.Tensor((M, N), BF16), WS: T.Tensor((WSU, BM, BN), F32)):
            with T.Kernel(NUM_SM, threads=384) as bid:
                sm1_as = T.alloc_shared((stages, BM, BK), BF16)
                sm0_ws = T.alloc_shared((stages, BN, BK), BF16)
                sm2_cs = T.alloc_shared((BM, BN), BF16)
                acc = T.alloc_fragment((BM, BN), F32)
                full = T.alloc_barrier([128] * stages)
                empty = T.alloc_barrier([256] * stages)

                if T.get_thread_binding() >= 256:
                    producer(A, W, sm1_as, sm0_ws, full, empty, bid)
                else:
                    consumer(sm1_as, sm0_ws, acc, full, empty, bid, WS, epi_residual, sm2_cs, R, C, sm2_cs, sm2_cs, sm2_cs, sm2_cs)

    elif epilogue == "swiglu":
        @T.prim_func
        def gemm_swiglu(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16), C: T.Tensor((M, NO), BF16),
                 WS: T.Tensor((WSU, BM, BN), F32)):
            with T.Kernel(NUM_SM, threads=384) as bid:
                sm1_as = T.alloc_shared((stages, BM, BK), BF16)
                sm0_ws = T.alloc_shared((stages, BN, BK), BF16)
                sm2_cs = T.alloc_shared((BM, BN), BF16)
                acc = T.alloc_fragment((BM, BN), F32)
                full = T.alloc_barrier([128] * stages)
                empty = T.alloc_barrier([256] * stages)

                if T.get_thread_binding() >= 256:
                    producer(A, W, sm1_as, sm0_ws, full, empty, bid)
                else:
                    consumer(sm1_as, sm0_ws, acc, full, empty, bid, WS, epi_swiglu, sm2_cs, C, sm2_cs, sm2_cs, sm2_cs, sm2_cs, sm2_cs)

    elif epilogue == "qk_norm_rope":
        @T.prim_func
        def gemm_qk_norm_rope(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16),
                 gq: T.Tensor((HD,), BF16), gk: T.Tensor((HD,), BF16),
                 cos: T.Tensor((M, ROT), BF16), sin: T.Tensor((M, ROT), BF16),
                 C: T.Tensor((M, N), BF16), WS: T.Tensor((WSU, BM, BN), F32)):
            with T.Kernel(NUM_SM, threads=384) as bid:
                sm1_as = T.alloc_shared((stages, BM, BK), BF16)
                sm0_ws = T.alloc_shared((stages, BN, BK), BF16)
                sm2_cs = T.alloc_shared((BM, BN), BF16)
                acc = T.alloc_fragment((BM, BN), F32)
                ss = T.alloc_fragment((BM,), F32)
                full = T.alloc_barrier([128] * stages)
                empty = T.alloc_barrier([256] * stages)

                if T.get_thread_binding() >= 256:
                    producer(A, W, sm1_as, sm0_ws, full, empty, bid)
                else:
                    consumer(sm1_as, sm0_ws, acc, full, empty, bid, WS, epi_qk, sm2_cs, ss, gq, gk, cos, sin, C)

    return {"plain": lambda: gemm_plain, "residual": lambda: gemm_residual, "swiglu": lambda: gemm_swiglu,
            "qk_norm_rope": lambda: gemm_qk_norm_rope}[epilogue]()


def gemm_tail_kernel(M: int, N: int, K: int, epilogue: str = "plain", BM: int = 128, BN: int = 256,
                     BK: int = 64, group: int = None, eps: float = 1e-6):
    """Finish the split tail: tile t's `splits` f32 partials (WS[t * splits + j],
    j in order, so the sum does not depend on timing) summed, then the same
    epilogue as `gemm_kernel`. One CTA per (tail band, mi, row chunk): CY
    chunks of RC = BM / CY rows, CY the largest power of two <= 8 that keeps
    the grid within 132 CTAs (scheduled/gemm.py `*_tail`; that TIR declares
    the module's 384 threads, of which these 256 participate)."""
    p = plan(M, N, K, epilogue, BM, BN, BK, group=group)
    group = p["group"]
    RC = BM // p["tail_cy"]
    mb, nb, rem, splits, whole, units = p["mb"], p["nb"], p["rem"], p["splits"], p["whole"], p["units"]
    NBT, B0, G0 = p["tail_bands"], p["band0"], mb // group - 1   # tail bands, first one, last M group
    assert rem > 0
    swiglu = epilogue == "swiglu"
    NO = N // 2 if swiglu else N
    BO = BN // 2 if swiglu else BN
    HD, ROT, HALF = 256, 64, 32
    NQG, NKH = 48, 4
    assert epilogue != "qk_norm_rope" or B0 >= NQG   # the tail holds k / v bands only (as scheduled/gemm.py asserts)

    @T.macro
    def gather(WS, acc, t, cy):
        """t: tail tile index (band - B0) * GROUP + mi."""
        T.clear(acc)
        for j in T.serial(splits):
            for i, c in T.Parallel(RC, BN):
                acc[i, c] = acc[i, c] + WS[t * splits + j, cy * RC + i, c]

    # residual / plain / swiglu: the TIR's ownership -- 384 threads launched,
    # the first 256 participate as (RC rows, TC = 256 / RC columns groups);
    # thread (row, col) holds CW contiguous columns of its row (for swiglu, CW
    # of the gate half and the same CW of the up half).
    TC = 256 // RC
    CW = (BO if swiglu else BN) // TC
    NK = CW // 4        # thread (row, col) holds columns (k * TC + col) * 4 + v, k < NK, v < 4

    if epilogue in ("residual", "plain", "swiglu"):
        @T.prim_func
        def tail(WS: T.Tensor((units, BM, BN), F32), R: T.Tensor((M, NO), BF16), C: T.Tensor((M, NO), BF16)):
            with T.Kernel(NBT, group, BM // RC, threads=384) as (tb, mi, cy):
                tx = T.get_thread_binding()
                acc = T.alloc_local((CW,), F32)
                acu = T.alloc_local((CW,), F32)
                ob = T.alloc_local((CW,), BF16)
                if tx < 256:
                    row = tx // TC
                    col = tx % TC
                    t = tb * group + mi
                    bm, bn = G0 * group + mi, B0 + tb
                    r = bm * BM + cy * RC + row
                    for k in T.unroll(NK):
                        for v in T.vectorized(4):
                            acc[k * 4 + v] = WS[t * splits, cy * RC + row, (k * TC + col) * 4 + v]
                        if swiglu:
                            for v in T.vectorized(4):
                                acu[k * 4 + v] = WS[t * splits, cy * RC + row, BO + (k * TC + col) * 4 + v]
                    for j in T.unroll(splits - 1):
                        for k in T.unroll(NK):
                            for v in T.vectorized(4):
                                acc[k * 4 + v] = acc[k * 4 + v] + WS[t * splits + 1 + j, cy * RC + row, (k * TC + col) * 4 + v]
                            if swiglu:
                                for v in T.vectorized(4):
                                    acu[k * 4 + v] = acu[k * 4 + v] + WS[t * splits + 1 + j, cy * RC + row, BO + (k * TC + col) * 4 + v]
                    for k in T.unroll(NK):
                        if epilogue == "residual":
                            for v in T.vectorized(4):
                                ob[k * 4 + v] = R[r, bn * BO + (k * TC + col) * 4 + v]
                        for v in T.unroll(4):
                            if epilogue == "residual":
                                ob[k * 4 + v] = T.cast(T.cast(ob[k * 4 + v], F32) + T.cast(T.cast(acc[k * 4 + v], BF16), F32), BF16)
                            elif epilogue == "plain":
                                ob[k * 4 + v] = T.cast(acc[k * 4 + v], BF16)
                            else:
                                g = T.cast(T.cast(acc[k * 4 + v], BF16), F32)
                                act = T.cast(T.call_extern(F32, "__fdividef", g, 1.0 + T.call_extern(F32, "__expf", -g)), BF16)
                                ob[k * 4 + v] = T.cast(T.cast(act, F32) * T.cast(T.cast(acu[k * 4 + v], BF16), F32), BF16)
                        for v in T.vectorized(4):
                            C[r, bn * BO + (k * TC + col) * 4 + v] = ob[k * 4 + v]
        return tail

    # qk_norm_rope (tir/qkv_tail_*.py): the same grid and the same interleaved
    # ownership; for a k (or q) head the row norm (sum over the thread's
    # columns, then over the TC lanes of the row), the normed row staged in
    # shared memory, rotate-half with thread (row, col) taking columns
    # (k * TC + col) * 4 + v of [0, HALF) and their partners HALF + ..., then
    # its own columns re-read and stored.
    QK = HALF // (4 * TC)

    @T.prim_func
    def tail(WS: T.Tensor((units, BM, BN), F32), gq: T.Tensor((HD,), BF16), gk: T.Tensor((HD,), BF16),
             cos: T.Tensor((M, ROT), BF16), sin: T.Tensor((M, ROT), BF16), C: T.Tensor((M, N), BF16)):
        with T.Kernel(NBT, group, BM // RC, threads=384) as (tb, mi, cy):
            tx = T.get_thread_binding()
            acc = T.alloc_local((CW,), F32)
            cb = T.alloc_local((CW,), BF16)
            red = T.alloc_local((1,), F32)
            lo = T.alloc_local((QK * 4,), F32)
            hi = T.alloc_local((QK * 4,), F32)
            sm0_ns = T.alloc_shared((RC, BN), BF16)
            bn = B0 + tb
            if tx < 256:
                row = tx // TC
                col = tx % TC
                t = tb * group + mi
                r = (G0 * group + mi) * BM + cy * RC + row
                for k in T.unroll(NK):
                    for v in T.vectorized(4):
                        acc[k * 4 + v] = WS[t * splits, cy * RC + row, (k * TC + col) * 4 + v]
                for j in T.unroll(splits - 1):
                    for k in T.unroll(NK):
                        for v in T.vectorized(4):
                            acc[k * 4 + v] = acc[k * 4 + v] + WS[t * splits + 1 + j, cy * RC + row, (k * TC + col) * 4 + v]
                for e in T.unroll(CW):
                    cb[e] = T.cast(acc[e], BF16)
                if bn >= NQG and bn < NQG + NKH:   # tir/qkv_tail_*.py: the tail holds k and v bands only; k heads normed
                    red[0] = 0.0
                    for e in T.unroll(CW):
                        red[0] = red[0] + T.cast(cb[e], F32) * T.cast(cb[e], F32)
                    for o in T.unroll(5):
                        if (TC >> (o + 1)) > 0:
                            red[0] = red[0] + T.shfl_xor(red[0], TC >> (o + 1), TC)
                    rs = T.rsqrt(red[0] / HD + eps)
                    for k in T.unroll(NK):
                        for v in T.unroll(4):
                            c0 = (k * TC + col) * 4 + v
                            gw = T.cast(gk[c0], F32)
                            cb[k * 4 + v] = T.cast(T.cast(cb[k * 4 + v], F32) * rs * (gw + 1.0), BF16)
                        for v in T.vectorized(4):
                            sm0_ns[row, (k * TC + col) * 4 + v] = cb[k * 4 + v]
                    T.sync_threads(1, 256)
                    for k in T.unroll(QK):
                        for v in T.unroll(4):
                            c0 = (k * TC + col) * 4 + v
                            lo[k * 4 + v] = T.cast(sm0_ns[row, c0], F32)
                            hi[k * 4 + v] = T.cast(sm0_ns[row, HALF + c0], F32)
                    for k in T.unroll(QK):
                        for v in T.unroll(4):
                            c0 = (k * TC + col) * 4 + v
                            sm0_ns[row, c0] = T.cast(T.cast(T.cast(lo[k * 4 + v] * T.cast(cos[r, c0], F32), BF16), F32)
                                                 + T.cast(T.cast(-hi[k * 4 + v] * T.cast(sin[r, c0], F32), BF16), F32), BF16)
                            sm0_ns[row, HALF + c0] = T.cast(T.cast(T.cast(hi[k * 4 + v] * T.cast(cos[r, HALF + c0], F32), BF16), F32)
                                                        + T.cast(T.cast(lo[k * 4 + v] * T.cast(sin[r, HALF + c0], F32), BF16), F32), BF16)
                    T.sync_threads(1, 256)
                    for k in T.unroll(NK):
                        for v in T.vectorized(4):
                            cb[k * 4 + v] = sm0_ns[row, (k * TC + col) * 4 + v]
                for k in T.unroll(NK):
                    for v in T.vectorized(4):
                        C[r, bn * BN + (k * TC + col) * 4 + v] = cb[k * 4 + v]
    return tail


# Shared memory as the TIR places it (tools/smem_plan.py): each buffer its own
# region, in name order (sm<k>_...: the TIR's address order).
SMEM_AS_TIR = {"tl.disable_shared_memory_reuse": True}


class Gemm:
    """A compiled GEMM: the main kernel, plus the tail kernel when there is a
    split tail. Called like `gemm_kernel` minus WS; `tail_args` picks the
    arguments the tail kernel needs from that call."""

    def __init__(self, M, N, K, epilogue, **kw):
        import torch

        self.plan = plan(M, N, K, epilogue, kw.get("BM", 128), kw.get("BN", 256), kw.get("BK", 64),
                         kw.get("split_tail", True), kw.get("group"))
        self.kernel = tilelang.compile(gemm_kernel(M, N, K, epilogue, **kw), out_idx=None, pass_configs=SMEM_AS_TIR)
        bm, bn = kw.get("BM", 128), kw.get("BN", 256)
        self.ws = torch.empty(max(self.plan["units"], 1), bm, bn, device="cuda", dtype=torch.float32)
        self.tail = None
        if self.plan["rem"]:
            tkw = {k: v for k, v in kw.items() if k in ("BM", "BN", "BK", "group", "eps")}
            self.tail = tilelang.compile(gemm_tail_kernel(M, N, K, epilogue, **tkw), out_idx=None, pass_configs=SMEM_AS_TIR)
        # main-kernel arguments after A, W that the tail kernel takes, in order
        self.tail_args = {"plain": (0, 0), "residual": (0, 1), "swiglu": (0, 0), "qk_norm_rope": (0, 1, 2, 3, 4)}[epilogue]

    def __call__(self, A, W, *rest):
        self.kernel(A, W, *rest, self.ws)
        if self.tail is not None:
            self.tail(self.ws, *(rest[i] for i in self.tail_args))

    def get_kernel_source(self):
        return self.kernel.get_kernel_source()


_CACHE = {}


def compile_gemm(M, N, K, epilogue="plain", **kw):
    key = (M, N, K, epilogue, tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = Gemm(M, N, K, epilogue, **kw)
    return _CACHE[key]
