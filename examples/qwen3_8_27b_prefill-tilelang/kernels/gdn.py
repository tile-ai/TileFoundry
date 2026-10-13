"""The chunked gated delta rule as two kernels (evidence/STEP3.md, cut P2, BV 64).

  intra      one CTA per (64-token chunk, value head), all chunks in parallel:
             gc = prefix sum of g in the chunk (f32)
             A  = strict_lower(bf16(beta k) k^T * exp(gc_i - gc_j))      f32
             T  = (I + A)^-1, by the published forward substitution, f32
             w  = bf16(T) @ bf16(beta k e^gc),  u = bf16(T) @ bf16(beta v)
             P  = bf16(scale * (q k^T) * exp(gc_i - gc_j)), lower incl. diagonal
             -- P does not depend on the state, so it leaves the serial pass.
  recur_out  one CTA per (value head, 64 state columns), chunks in order:
             v_new = u - w S;  o = scale e^gc (q S) + P v_new;
             S <- e^gc_last S + k^T (v_new e^(gc_last - gc))
             S stays f32 in registers; its bf16 copy feeds the tensor cores.

Precision follows FLA's chunk_gated_delta_rule (bf16 tensor-core products,
f32 accumulation and decays), the implementation the measured SGLang baseline
runs; model.py states the f32 published arithmetic and `check` holds these to
it within the bounds the landings imply.
"""

import tilelang
import tilelang.language as T

BF16, F32 = "bfloat16", "float32"
HK, HV, D, CH = 16, 48, 128, 64
WUP = 2 * D + CH        # intra's packed output per (token, value head): w | u | P row
VPK = HV // HK
SCALE = D ** -0.5


# tir/delta_rule_intra_wup_{n}.py (the same at 1024, 4096, 8192) and intra_g's gsm
OFF = {"gsm": 0, "ost": 0, "gb": 16384, "xw1s": 16384, "ks": 17408, "qs": 33792, "ps": 33792,
       "asm": 41984, "tb": 41984, "vs": 50176, "xu1s": 50176, "ost_1": 50176, "maa": 58624,
       "mbb": 62848, "xu0s": 66560, "xw0s": 66560, "value_42": 67072}  # value_42: Y (the HIR leaves it unnamed)
POOL_BYTES = 74752
ROWB = 128                      # bytes per pool row (64 bf16)

PRELUDE = r"""
__device__ __forceinline__ float tf_ld(const void* base, int off) {
  return *reinterpret_cast<const float*>(reinterpret_cast<const char*>(base) + off);
}
__device__ __forceinline__ void tf_st(void* base, int off, float v) {
  *reinterpret_cast<float*>(reinterpret_cast<char*>(base) + off) = v;
}
"""


def R(name, chunk=0):
    """First pool row of a bf16 tile's 64-column chunk."""
    assert OFF[name] % 1024 == 0
    return OFF[name] // ROWB + 64 * chunk


def intra_kernel(M: int, skip: tuple = (), IOTA: str = "int32"):
    """tir/delta_rule_intra_g_{M}.py + tir/delta_rule_intra_wup_{M}.py
    (scheduled/gdn.py `DeltaRuleIntra{M}`), the two device functions of one
    invocation, in one launch: grid (chunks, value heads), 128 threads.

      intra_g    threads 0..63: gc_i = g_0 + .. + g_i in order -> Go and smem.
      intra_wup  K, Q, V by TMA (warp 0); K K^T, Q K^T (m64n64k16 WGMMA);
                 in the fragment's ownership, decay = exp(gc_i - gc_j),
                 P = bf16(scale q k^T decay) lower incl. the diagonal (stored
                 at once), L = -strict_lower(beta k k^T decay) in f32 smem.
                 T = (I - L)^-1: phase 1, threads 0..63, thread (b, j) column
                 j of diagonal block b, rows in order (x_i = L_ij + sum_k
                 L_ik x_k); phase 2a, both 32-blocks, thread (ti, tj) rows ti,
                 ti + 8 of column tj, computed transposed as stored above the
                 diagonal; phase 2b, the lower-left 32-block, f32 products
                 per thread (no f32 / tf32 MMA atom in the TIR's facts).
                 T in bf16 in the WGMMA fragment's ownership; w^T = X_w^T T^T
                 and u^T = X_u^T T^T on the tensor cores, staged transposed
                 and stored by TMA into the packed [w | u | P] rows.
    Shared memory as the TIR places it (tools/smem_plan.py, `OFF`): one (584, 64) bf16
    allocation; each bf16 tile is a row slice of it at the TIR's byte offset (every one a
    multiple of 1024 B = 8 rows, so TileLang's SW128 swizzle of the allocation is the TIR's
    swizzle of the tile; a (64, 128) tile is two 64-column chunks 8 KB apart, the TIR's
    layout); each f32 tile is read and written at its TIR byte address in the same
    allocation (`tf_ld` / `tf_st`), with the TIR's row-major layout and padding. Tiles share
    bytes in turn as the TIR's plan lets them; a barrier separates the last read of a tile
    from the first write of the next one on its bytes.
    `skip` drops phases for timing experiments only (results are then wrong)."""
    C = M // CH
    A0, M0, B0, Y0, G0, S0 = OFF["asm"], OFF["maa"], OFF["mbb"], OFF["value_42"], OFF["gb"], OFF["gsm"]
    KS0, KS1, QS0, QS1 = R("ks"), R("ks", 1), R("qs"), R("qs", 1)
    VS0, VS1, PS, TB, X0, XU1, XW1 = R("vs"), R("vs", 1), R("ps"), R("tb"), R("xu0s"), R("xu1s"), R("xw1s")
    OS0, OS1, OW0, OW1 = R("ost"), R("ost", 1), R("ost_1"), R("ost_1", 1)

    @T.prim_func
    def intra(Q: T.Tensor((M, HK, D), BF16), K: T.Tensor((M, HK, D), BF16), V: T.Tensor((M, HV, D), BF16),
              G: T.Tensor((M, HV), F32), Beta: T.Tensor((M, HV), BF16), Iota: T.Tensor((CH,), IOTA),
              WU: T.Tensor((M, HV * WUP), BF16), Go: T.Tensor((HV, M), F32)):
        with T.Kernel(C, HV, threads=128, prelude=PRELUDE) as (c, h):
            kh = h // VPK
            t0 = c * CH
            pool = T.alloc_shared((POOL_BYTES // ROWB, 64), BF16)
            ldbar = T.alloc_barrier([128, 128])
            kk = T.alloc_fragment((CH, CH), F32)
            qk = T.alloc_fragment((CH, CH), F32)
            o0 = T.alloc_fragment((CH, 64), F32)
            o1 = T.alloc_fragment((CH, 64), F32)
            xb = T.alloc_local((16,), F32)
            xt = T.alloc_local((2,), F32)
            yv = T.alloc_local((8,), F32)
            ab = T.alloc_local((8,), F32)
            av = T.alloc_local((4,), F32)
            bv = T.alloc_local((2,), F32)
            acc = T.alloc_local((1,), F32)
            sc = T.alloc_local((2,), F32)
            xw = T.alloc_local((32,), BF16)
            ir = T.alloc_fragment((CH,), IOTA)
            jc = T.alloc_fragment((CH,), IOTA)
            tx = T.get_thread_binding()
            base = T.address_of(pool[0, 0])

            # K, Q by TMA, each as two 64-column chunks 8 KB apart (the TIR's ks / qs layout)
            T.tma_copy(K[t0:t0 + CH, kh, 0:64], pool[KS0:KS0 + CH, :], barrier=ldbar[0])
            T.tma_copy(K[t0:t0 + CH, kh, 64:128], pool[KS1:KS1 + CH, :], barrier=ldbar[0])
            T.tma_copy(Q[t0:t0 + CH, kh, 0:64], pool[QS0:QS0 + CH, :], barrier=ldbar[0])
            T.tma_copy(Q[t0:t0 + CH, kh, 64:128], pool[QS1:QS1 + CH, :], barrier=ldbar[0])
            T.mbarrier_arrive(ldbar[0])
            # intra_g: the chunk's g staged in smem (the TIR's gsm; thread i: g_i), then thread i of the
            # first 64 sums g_j (jc <= ir) over the chunk, j ascending
            if tx < CH:
                T.evaluate(T.call_extern("handle", "tf_st", base, S0 + 4 * tx, G[t0 + tx, h]))
            T.sync_threads()
            if tx < CH:
                acc[0] = 0.0
                for j in T.unroll(CH):
                    acc[0] = acc[0] + T.call_extern(F32, "tf_ld", base, S0 + 4 * j) * T.cast(Iota[j] <= Iota[tx], F32)
                Go[h, t0 + tx] = acc[0]
            T.sync_threads()
            # intra_wup: gc and beta of the chunk staged in f32 smem (the TIR's gb: (4, 64), cleared,
            # row 0 gc, row 1 beta; threads 0..63, row i)
            for e in T.unroll(2):
                T.evaluate(T.call_extern("handle", "tf_st", base, G0 + 4 * (tx + 128 * e), T.float32(0)))
            T.sync_threads()
            if tx < CH:
                T.evaluate(T.call_extern("handle", "tf_st", base, G0 + 4 * tx, Go[h, t0 + tx]))
                T.evaluate(T.call_extern("handle", "tf_st", base, G0 + 4 * CH + 4 * tx, T.cast(Beta[t0 + tx, h], F32)))
            T.mbarrier_wait_parity(ldbar[0], 0)
            T.gemm(pool[KS0:KS0 + CH, :], pool[KS0:KS0 + CH, :], kk, transpose_B=True, clear_accum=True)
            T.gemm(pool[KS1:KS1 + CH, :], pool[KS1:KS1 + CH, :], kk, transpose_B=True)
            T.gemm(pool[QS0:QS0 + CH, :], pool[KS0:KS0 + CH, :], qk, transpose_B=True, clear_accum=True)
            T.gemm(pool[QS1:QS1 + CH, :], pool[KS1:KS1 + CH, :], qk, transpose_B=True)
            for i in T.Parallel(CH):
                ir[i] = Iota[i]
            for j in T.Parallel(CH):
                jc[j] = Iota[j]
            T.sync_threads()          # asm's bytes include qs's second chunk, read by the products above
            for e in T.serial((CH * (CH + 1) + 127) // 128):
                if tx + 128 * e < CH * (CH + 1):
                    T.evaluate(T.call_extern("handle", "tf_st", base, A0 + 4 * (tx + 128 * e), T.float32(0)))
            T.sync_threads()
            for i, j in T.Parallel(CH, CH):
                dec = T.exp(T.call_extern(F32, "tf_ld", base, G0 + 4 * i) - T.call_extern(F32, "tf_ld", base, G0 + 4 * j))
                pool[PS + i, j] = T.cast(T.if_then_else(ir[i] - jc[j] >= 0, qk[i, j] * dec * SCALE, 0.0), BF16)
                T.evaluate(T.call_extern("handle", "tf_st", base, A0 + 4 * (i * (CH + 1) + j), T.if_then_else(
                    ir[i] - jc[j] > 0, -(kk[i, j] * T.call_extern(F32, "tf_ld", base, G0 + 4 * CH + 4 * i) * dec), 0.0)))
            T.sync_threads()
            ti = tx // 16
            tj = tx % 16
            if "sub" not in skip:
                # phase 1: the four 16 x 16 diagonal blocks
                if tx < 64 and "p1" not in skip:
                    o = (tx // 16) * 16
                    j = tx % 16
                    for i in T.unroll(16):
                        xb[i] = 0.0
                    for i in T.unroll(16):
                        acc[0] = 0.0
                        for k in T.unroll(16):
                            acc[0] = acc[0] + T.call_extern(F32, "tf_ld", base, A0 + 4 * ((o + i) * (CH + 1) + o + k)) * xb[k]
                        xb[i] = T.call_extern(F32, "tf_ld", base, A0 + 4 * ((o + i) * (CH + 1) + o + j)) + acc[0]
                    T.sync_warp()
                    for i in T.unroll(16):
                        T.evaluate(T.call_extern("handle", "tf_st", base, A0 + 4 * ((o + i) * (CH + 1) + o + j), xb[i]))
                T.sync_threads()
                # phase 2a: X^T = L21^T + N11^T L21^T, then T21^T = X^T + X^T N22^T, stored above the diagonal
                for sb in T.unroll(0 if "p2a" in skip else 2):
                    o = 32 * sb
                    for e in T.unroll(2):
                        acc[0] = 0.0
                        for k in T.unroll(16):
                            acc[0] = acc[0] + T.call_extern(F32, "tf_ld", base, A0 + 4 * ((o + k) * (CH + 1) + o + tj)) \
                                * T.call_extern(F32, "tf_ld", base, A0 + 4 * ((o + 16 + 8 * e + ti) * (CH + 1) + o + k))
                        xt[e] = T.call_extern(F32, "tf_ld", base, A0 + 4 * ((o + 16 + 8 * e + ti) * (CH + 1) + o + tj)) + acc[0]
                    for e in T.unroll(2):
                        T.evaluate(T.call_extern("handle", "tf_st", base, A0 + 4 * ((o + tj) * (CH + 1) + o + 16 + 8 * e + ti), xt[e]))
                    T.sync_threads()
                    for e in T.unroll(2):
                        acc[0] = 0.0
                        for k in T.unroll(16):
                            acc[0] = acc[0] + T.call_extern(F32, "tf_ld", base, A0 + 4 * ((o + tj) * (CH + 1) + o + 16 + k)) \
                                * T.call_extern(F32, "tf_ld", base, A0 + 4 * ((o + 16 + 8 * e + ti) * (CH + 1) + o + 16 + k))
                        yv[e] = xt[e] + acc[0]
                    T.sync_threads()
                    for e in T.unroll(2):
                        T.evaluate(T.call_extern("handle", "tf_st", base, A0 + 4 * ((o + tj) * (CH + 1) + o + 16 + 8 * e + ti), yv[e]))
                    T.sync_threads()
                if "p2b" not in skip:
                    # phase 2b: M_AA, M_BB (strictly lower parts of the 32 x 32 diagonal blocks of T),
                    # cleared first as the TIR clears them (and Y)
                    for e in T.serial((32 * 33 + 127) // 128):
                        if tx + 128 * e < 32 * 33:
                            T.evaluate(T.call_extern("handle", "tf_st", base, M0 + 4 * (tx + 128 * e), T.float32(0)))
                            T.evaluate(T.call_extern("handle", "tf_st", base, B0 + 4 * (tx + 128 * e), T.float32(0)))
                            T.evaluate(T.call_extern("handle", "tf_st", base, Y0 + 4 * (tx + 128 * e), T.float32(0)))
                    T.sync_threads()
                    for e in T.unroll(8):
                        i = (tx + 128 * e) // 32
                        j = (tx + 128 * e) % 32
                        T.evaluate(T.call_extern("handle", "tf_st", base, M0 + 4 * (i * 33 + j), T.if_then_else(
                            i // 16 == j // 16, T.call_extern(F32, "tf_ld", base, A0 + 4 * (i * (CH + 1) + j)),
                            T.if_then_else(i >= 16 and j < 16, T.call_extern(F32, "tf_ld", base, A0 + 4 * (j * (CH + 1) + i)), 0.0))))
                        T.evaluate(T.call_extern("handle", "tf_st", base, B0 + 4 * (i * 33 + j), T.if_then_else(
                            i // 16 == j // 16, T.call_extern(F32, "tf_ld", base, A0 + 4 * ((32 + i) * (CH + 1) + 32 + j)),
                            T.if_then_else(i >= 16 and j < 16,
                                           T.call_extern(F32, "tf_ld", base, A0 + 4 * ((32 + j) * (CH + 1) + 32 + i)), 0.0))))
                    T.sync_threads()
                    # Y = L_BA + L_BA M_AA
                    for r in T.unroll(8):
                        ab[r] = 0.0
                    for k in T.serial(32):
                        for e in T.unroll(4):
                            av[e] = T.call_extern(F32, "tf_ld", base, A0 + 4 * ((32 + 8 * e + ti) * (CH + 1) + k))
                        for f in T.unroll(2):
                            bv[f] = T.call_extern(F32, "tf_ld", base, M0 + 4 * (k * 33 + 16 * f + tj))
                        for e in T.unroll(4):
                            for f in T.unroll(2):
                                ab[e * 2 + f] = ab[e * 2 + f] + av[e] * bv[f]
                    for e in T.unroll(4):
                        for f in T.unroll(2):
                            yv[e * 2 + f] = T.call_extern(F32, "tf_ld", base, A0 + 4 * ((32 + 8 * e + ti) * (CH + 1) + 16 * f + tj)) \
                                + ab[e * 2 + f]
                            T.evaluate(T.call_extern("handle", "tf_st", base, Y0 + 4 * ((8 * e + ti) * 33 + 16 * f + tj), yv[e * 2 + f]))
                    T.sync_threads()
                    # T_BA^T = Y^T + Y^T M_BB^T, stored above the diagonal
                    for r in T.unroll(8):
                        ab[r] = 0.0
                    for k in T.serial(32):
                        for f in T.unroll(2):
                            bv[f] = T.call_extern(F32, "tf_ld", base, Y0 + 4 * (k * 33 + 16 * f + tj))
                        for e in T.unroll(4):
                            av[e] = T.call_extern(F32, "tf_ld", base, B0 + 4 * ((8 * e + ti) * 33 + k))
                        for e in T.unroll(4):
                            for f in T.unroll(2):
                                ab[e * 2 + f] = ab[e * 2 + f] + bv[f] * av[e]
                    for f in T.unroll(2):
                        for e in T.unroll(4):
                            T.evaluate(T.call_extern("handle", "tf_st", base, A0 + 4 * ((16 * f + tj) * (CH + 1) + 32 + 8 * e + ti),
                                                     yv[e * 2 + f] + ab[e * 2 + f]))
                    T.sync_threads()
            # T = I + N in bf16, in the WGMMA fragment's ownership: d = ir - jc, db = ir // 16 - jc // 16
            # from the iota table; N inside 16-blocks below the diagonal, transposed storage across them
            # tm in kk's registers (dead since L), then bf16 into tb
            for i, j in T.Parallel(CH, CH):
                kk[i, j] = T.if_then_else(ir[i] - jc[j] == 0, 1.0, T.if_then_else(ir[i] - jc[j] > 0, T.if_then_else(
                    ir[i] // 16 - jc[j] // 16 == 0, T.call_extern(F32, "tf_ld", base, A0 + 4 * (i * (CH + 1) + j)),
                    T.call_extern(F32, "tf_ld", base, A0 + 4 * (j * (CH + 1) + i))), 0.0))
            T.sync_threads()          # tb's bytes are asm's first 8 KB
            T.copy(kk, pool[TB:TB + CH, :])
            T.sync_threads()
            # V lands where the TIR loads it: after T, before the products that read it. vs's bytes held
            # asm / maa / mbb (generic-proxy writes); the TMA load writes them next
            T.fence_proxy_async()
            T.tma_copy(V[t0:t0 + CH, h, 0:64], pool[VS0:VS0 + CH, :], barrier=ldbar[1])
            T.tma_copy(V[t0:t0 + CH, h, 64:128], pool[VS1:VS1 + CH, :], barrier=ldbar[1])
            T.mbarrier_arrive(ldbar[1])
            T.sync_threads()
            if "wu" not in skip:
                # u = T X_u, X_u = bf16(beta v), then w = T X_w, X_w = bf16(beta k e^gc) (the TIR's
                # order), one 64-wide X chunk at a time: an m64n64 product (A = T K-major, B = the
                # chunk MN-major); thread (ti, tj) makes X rows ti + 8 e, columns tj * 4 ... Each tile
                # where the TIR puts it: X chunk 0 in xu0s / xw0s, X_u's chunk 1 over V's chunk 0 (read
                # already), X_w's chunk 1 at xw1s; u staged in ost, w in ost_1 (V's bytes); P, w, u
                # stored at the end.
                T.mbarrier_wait_parity(ldbar[1], 0)
                for e in T.unroll(8):
                    r = 8 * e + ti
                    sc[1] = T.call_extern(F32, "tf_ld", base, G0 + 4 * CH + 4 * r)
                    for f in T.vectorized(4):
                        pool[X0 + r, tj * 4 + f] = T.cast(T.cast(pool[VS0 + r, tj * 4 + f], F32) * sc[1], BF16)
                T.sync_threads()
                T.gemm(pool[TB:TB + CH, :], pool[X0:X0 + CH, :], o0, clear_accum=True)
                T.copy(o0, pool[OS0:OS0 + CH, :])
                for e in T.unroll(8):
                    r = 8 * e + ti
                    sc[1] = T.call_extern(F32, "tf_ld", base, G0 + 4 * CH + 4 * r)
                    for f in T.vectorized(4):
                        pool[XU1 + r, tj * 4 + f] = T.cast(T.cast(pool[VS1 + r, tj * 4 + f], F32) * sc[1], BF16)
                T.sync_threads()
                T.gemm(pool[TB:TB + CH, :], pool[XU1:XU1 + CH, :], o1, clear_accum=True)
                T.copy(o1, pool[OS1:OS1 + CH, :])
                for e in T.unroll(8):
                    r = 8 * e + ti
                    sc[0] = T.call_extern(F32, "tf_ld", base, G0 + 4 * r)
                    sc[1] = T.call_extern(F32, "tf_ld", base, G0 + 4 * CH + 4 * r)
                    for f in T.vectorized(4):
                        pool[X0 + r, tj * 4 + f] = T.cast(T.cast(pool[KS0 + r, tj * 4 + f], F32) * sc[1] * T.exp(sc[0]), BF16)
                T.sync_threads()
                T.gemm(pool[TB:TB + CH, :], pool[X0:X0 + CH, :], o0, clear_accum=True)
                T.copy(o0, pool[OW0:OW0 + CH, :])
                # xw1s's first 1024 bytes are gb: every thread reads its gc / beta before any writes
                for e in T.unroll(8):
                    r = 8 * e + ti
                    sc[0] = T.call_extern(F32, "tf_ld", base, G0 + 4 * r)
                    sc[1] = T.call_extern(F32, "tf_ld", base, G0 + 4 * CH + 4 * r)
                    for f in T.unroll(4):
                        xw[e * 4 + f] = T.cast(T.cast(pool[KS1 + r, tj * 4 + f], F32) * sc[1] * T.exp(sc[0]), BF16)
                T.sync_threads()
                for e in T.unroll(8):
                    for f in T.vectorized(4):
                        pool[XW1 + 8 * e + ti, tj * 4 + f] = xw[e * 4 + f]
                T.sync_threads()
                T.gemm(pool[TB:TB + CH, :], pool[XW1:XW1 + CH, :], o1, clear_accum=True)
                T.copy(o1, pool[OW1:OW1 + CH, :])
            T.copy(pool[PS:PS + CH, :], WU[t0:t0 + CH, h * WUP + 2 * D:h * WUP + 2 * D + CH])
            T.copy(pool[OW0:OW0 + CH, :], WU[t0:t0 + CH, h * WUP:h * WUP + 64])
            T.copy(pool[OW1:OW1 + CH, :], WU[t0:t0 + CH, h * WUP + 64:h * WUP + D])
            T.copy(pool[OS0:OS0 + CH, :], WU[t0:t0 + CH, h * WUP + D:h * WUP + D + 64])
            T.copy(pool[OS1:OS1 + CH, :], WU[t0:t0 + CH, h * WUP + D + 64:h * WUP + 2 * D])

    return intra



def recur_out_kernel(M: int, BV: int = 64, skip: tuple = ()):
    """`skip` drops parts for timing probes only (results are then wrong):
    "store", "pv", "s_gemm", "s_copy", "v_gemm" (tools/time_recur.py).

    Warp-specialized: the state recurrence is the serial path, so it gets a
    warpgroup of its own; the read-out and the loads run beside it.

      WG0 (0..127)    v_new = u - w S  -> Vn[c % 3];  S <- e^gl S + K^T (v_new e^(gl - gc))
                      -> Sb[(c + 1) % 3]
      WG1 (128..255)  o = scale e^gc (q S_c) + P v_new   (S_c = Sb[c % 3])
      WG2 (256..383)  TMA: w, k, q, u, P, gc of chunk c into slot c % 2

    Sb and Vn are triple-buffered: WG0 writes S_(c+1) into the slot WG1
    finished reading at chunk c - 2.
    """
    C = M // CH
    NV = D // BV
    NS = 2   # input slots
    NB = 3   # state / v_new slots

    @T.prim_func
    def recur_out(Q: T.Tensor((M, HK, D), BF16), K: T.Tensor((M, HK, D), BF16),
                  WU: T.Tensor((M, HV * WUP), BF16), Gc: T.Tensor((HV, M), F32),
                  O: T.Tensor((M, HV, D), BF16)):
        with T.Kernel(NV, HV, threads=384) as (vb, h):
            kh = h // VPK
            v0 = vb * BV
            sm03_qs = T.alloc_shared((NS, CH, D), BF16)
            sm02_ks = T.alloc_shared((NS, CH, D), BF16)
            sm01_ws = T.alloc_shared((NS, CH, D), BF16)
            sm05_us = T.alloc_shared((NS, CH, BV), BF16)
            sm06_ps = T.alloc_shared((NS, CH, CH), BF16)
            sm00_sb = T.alloc_shared((NB, D, BV), BF16)
            sm04_vnb = T.alloc_shared((NB, CH, BV), BF16)
            sm09_vd = T.alloc_shared((CH, BV), BF16)
            # the TIR's eds (3 slots), eqs (3 slots) and gs (2 slots) lie back to back
            # from one offset; one buffer keeps them there (TileLang would align the
            # TMA target gs to 1024 bytes on its own)
            sm08_e = T.alloc_shared((2 * NB * CH + NS * CH,), F32)
            S = T.alloc_fragment((D, BV), F32)
            vn = T.alloc_fragment((CH, BV), F32)
            qs = T.alloc_fragment((CH, BV), F32)
            sm07_os = T.alloc_shared((2, CH, BV), BF16)
            full = T.alloc_barrier([128] * NS)
            empty = T.alloc_barrier([256] * NS)
            vready = T.alloc_barrier([128] * NB)
            vfree = T.alloc_barrier([128] * NB)
            tx = T.get_thread_binding()
            if tx < 128:
                # WG0: the state.
                T.clear(S)
                T.copy(S, sm00_sb[0, :, :])
                T.fence_proxy_async()
                for c in T.serial(C):
                    st = c % NS
                    T.mbarrier_wait_parity(full[st], (c // NS) % 2)
                    # Per-row decays, once per chunk: e^(gl - g_i) for the state
                    # update, scale e^(g_i) for WG1's read-out, e^gl.
                    if tx < CH:
                        sm08_e[(c % NB) * CH + tx] = T.exp(sm08_e[2 * NB * CH + st * CH + CH - 1] - sm08_e[2 * NB * CH + st * CH + tx])
                        sm08_e[NB * CH + (c % NB) * CH + tx] = T.exp(sm08_e[2 * NB * CH + st * CH + tx]) * SCALE
                    if "v_gemm" not in skip:
                        T.wgmma_gemm(sm01_ws[st, :, :], sm00_sb[c % NB, :, :], vn, clear_accum=True)
                    # the decay of S does not need v_new: it runs under the WGMMA
                    el = T.exp(sm08_e[2 * NB * CH + st * CH + CH - 1])
                    for i, j in T.Parallel(D, BV):
                        S[i, j] = S[i, j] * el
                    T.wait_wgmma(0)
                    for i, j in T.Parallel(CH, BV):
                        vn[i, j] = T.cast(sm05_us[st, i, j], F32) - vn[i, j]
                    if c >= 2:
                        T.mbarrier_wait_parity(vfree[(c + 1) % NB], ((c - 2) // NB) % 2)
                    for i, j in T.Parallel(CH, BV):
                        sm04_vnb[c % NB, i, j] = T.cast(vn[i, j], BF16)
                        sm09_vd[i, j] = T.cast(vn[i, j] * sm08_e[(c % NB) * CH + i], BF16)
                    T.fence_proxy_async()
                    T.mbarrier_arrive(vready[c % NB])
                    if "s_gemm" not in skip:
                        T.gemm(sm02_ks[st, :, :], sm09_vd, S, transpose_A=True)
                    T.mbarrier_arrive(empty[st])
                    if "s_copy" not in skip:
                        T.copy(S, sm00_sb[(c + 1) % NB, :, :])
                    T.fence_proxy_async()
            elif tx < 256:
                # WG1: the read-out.
                for c in T.serial(C):
                    st = c % NS
                    T.mbarrier_wait_parity(full[st], (c // NS) % 2)
                    T.mbarrier_wait_parity(vready[c % NB], (c // NB) % 2)
                    T.gemm(sm03_qs[st, :, :], sm00_sb[c % NB, :, :], qs, clear_accum=True)
                    for i, j in T.Parallel(CH, BV):
                        qs[i, j] = qs[i, j] * sm08_e[NB * CH + (c % NB) * CH + i]
                    if "pv" not in skip:
                        T.gemm(sm06_ps[st, :, :], sm04_vnb[c % NB, :, :], qs)
                    T.mbarrier_arrive(vfree[c % NB])
                    T.mbarrier_arrive(empty[st])
                    if "store" not in skip:
                        # staged in shared memory, stored by TMA (a direct
                        # fragment store scatters 4-byte writes over rows
                        # HV * D apart and held WG1 -- and through vfree WG0
                        # -- back: evidence/STEP4.md row 12)
                        T.copy(qs, sm07_os[c % 2, :, :])
                        T.copy(sm07_os[c % 2, :, :], O[c * CH:(c + 1) * CH, h, v0:v0 + BV])
            else:
                # WG2: the loads.
                for c in T.serial(C):
                    st = c % NS
                    t0 = c * CH
                    T.mbarrier_wait_parity(empty[st], ((c // NS) % 2) ^ 1)
                    T.tma_copy(WU[t0:t0 + CH, h * WUP:h * WUP + D], sm01_ws[st, :, :], barrier=full[st])
                    T.tma_copy(K[t0:t0 + CH, kh, :], sm02_ks[st, :, :], barrier=full[st])
                    T.tma_copy(Q[t0:t0 + CH, kh, :], sm03_qs[st, :, :], barrier=full[st])
                    T.tma_copy(WU[t0:t0 + CH, h * WUP + D + v0:h * WUP + D + v0 + BV], sm05_us[st, :, :], barrier=full[st])
                    T.tma_copy(WU[t0:t0 + CH, h * WUP + 2 * D:h * WUP + 2 * D + CH], sm06_ps[st, :, :], barrier=full[st])
                    T.tma_copy(Gc[h, t0:t0 + CH], sm08_e[2 * NB * CH + st * CH:2 * NB * CH + st * CH + CH], barrier=full[st])
                    T.mbarrier_arrive(full[st])

    return recur_out


_CACHE = {}


def compiled(name, *args, **kw):
    key = (name, args, tuple(sorted(kw.items())))
    if key not in _CACHE:
        fn = {"intra": intra_kernel, "recur_out": recur_out_kernel}[name]
        # shared memory as the TIR places it (tools/smem_plan.py): recur_out, each buffer its own
        # region in name order (sm<k>_...: the TIR's address order); intra, one allocation
        cfg = {"tl.disable_shared_memory_reuse": True}
        _CACHE[key] = tilelang.compile(fn(*args, **kw), out_idx=None, pass_configs=cfg)
    return _CACHE[key]
