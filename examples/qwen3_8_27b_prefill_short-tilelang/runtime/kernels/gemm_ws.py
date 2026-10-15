"""Hand warp-specialized GEMM kernels: C[M, N] = A[M, K] @ W[N, K]^T (bf16 in, f32 acc).

Same operands, epilogues and rounding as runtime/kernels/gemm.py gemm_kernel; the per-element
K order is the same (one f32 wgmma accumulator, k16 steps ascending), so outputs are bitwise
equal to gemm_kernel with the same epilogue and the same split (tested on random data at
T = 128 / 256 / 512 / 1024 for every config in WS_CONFIGS; GEMM_WS_NOTES.md).

Roles (384 threads, 1 CTA per SM): warpgroup 0 = producer (warp 0 issues all TMA loads; warp 1
is the optional IO warp), warpgroups 1-2 = consumers (cooperative: one BM=128 tile, warpgroup g
owns rows [64g, 64g+64), FullRow, explicit T.wgmma_gemm + T.wait_wgmma(0)). Ring: `stages` slots
of A (BM x BK) and W (BN x BK) (swiglu2: W_gate and W_up); full barriers (count 32: the producer
warp arrives after expect_tx), empty barriers (count 8: lane 0 of each consumer warp).

Options used by WS_CONFIGS:
  grid > 0  persistent: `grid` CTAs walk the tile list (M fastest, so the CTAs that run at the
            same time share one W tile); the producer runs ahead into the next tile while the
            consumers do the epilogue; the epilogue's TMA store is waited for only before the
            staging tile is written again.
  pk        epilogue casts as packed bf16x2 conversions (T.copy fragment -> fragment) and a
            non-volatile rcp.approx.ftz: same roundings, F2F (quarter rate) removed from SASS.
  amerge    pass config tl.enable_aggressive_shared_memory_merge: the staging tile aliases the
            A ring (one tile per CTA only), which frees smem for one more ring stage.
  io        producer warp 1 TMA-loads the residual tile R into the staging tile before the
            mainloop ends and issues the output TMA store (one tile per CTA: R latency hidden).
  csk       split_k = 2 over a 2-CTA cluster (rank r = K half r): rank 1 bulk-copies its f32
            partial tile into rank 0's smem (DSMEM, cluster mbarrier), rank 0 adds
            (0 + P0) + P1 (splitk_reduce_kernel's order) and runs the residual epilogue: no
            separate reduce kernel, no f32 partials in HBM; bitwise equal to split-K + reduce.
  direct    epilogue writes fragments straight to global (no staging tile).
Experimental options (measured, not used; GEMM_WS_NOTES.md): CL / mc (TMA multicast of A or W
inside a cluster), wait1, wbox (W tile as several TMA boxes), pdl (W prefetch before
griddepcontrol.wait), debug (nomma / noa / now / wonly / noepi: timing decomposition only).

Call convention (all tensors bf16, contiguous): store k(A, W, C); swiglu2 k(A, Wg, Wu, O);
residual k(A, W, R, C) (also for csk); splitk k(A, W, P) + gemm.splitk_reduce_kernel.
"""

from dataclasses import dataclass
from functools import lru_cache

import tilelang
import tilelang.language as T

BF16, F32 = "bfloat16", "float"
_POLICY = T.GemmWarpPolicy.FullRow
NSM = 132


@dataclass(frozen=True)
class WsConfig:
    BN: int = 256
    BK: int = 64
    stages: int = 4
    CL: int = 1           # cluster size (TMA multicast of one operand inside the cluster)
    mc: str = "A"         # "A": cluster along N, A tile multicast; "W": cluster along M, W tile(s) multicast
    grid: int = 0         # persistent grid (CTAs); 0 = one tile per CTA
    wait1: bool = False   # keep one wgmma batch in flight, release the previous slot
    split_k: int = 1
    direct: bool = False  # epilogue writes fragments straight to global (no smem staging tile)
    amerge: bool = False  # pass config tl.enable_aggressive_shared_memory_merge
    pdl: bool = False     # programmatic dependent launch: W of the first ring slots is loaded before griddepcontrol.wait
    csk: bool = False     # split_k=2 over a 2-CTA cluster: rank 1 sends its f32 partial to rank 0 (DSMEM)
    pk: bool = False      # epilogue casts as packed bf16x2 conversions (T.copy), non-volatile rcp.approx
    io: bool = False      # producer warp 1 loads R (residual) into the staging tile and issues the TMA stores
    wbox: int = 0         # load each W tile as boxes of at most `wbox` rows (0 = one box)
    debug: str = ""       # experiments only: "nomma" (no wgmma), "noa" (no A loads), "now" (no W loads)
    BM: int = 128

    def tag(self) -> str:
        p = f"_p{self.grid}" if self.grid else ""
        sk = f"_sk{self.split_k}" if self.split_k > 1 else ""
        d = ("_am" if self.amerge else "") + ("_pdl" if self.pdl else "") + ("_csk" if self.csk else "") + ("_pk" if self.pk else "") + ("_io" if self.io else "") + (f"_wb{self.wbox}" if self.wbox else "") + ("_d" if self.direct else "") + (f"_{self.debug}" if self.debug else "")
        c = f"_c{self.CL}{self.mc}" if self.CL > 1 else ""
        return f"ws{self.BM}x{self.BN}x{self.BK}_s{self.stages}{c}{p}{'_w1' if self.wait1 else ''}{sk}{d}"


# rcp.approx.ftz.f32 as tl::fast_rcp emits it, but not volatile, so ptxas may schedule it
_PRELUDE = '''
__device__ __forceinline__ float gws_rcp_approx(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
'''


def _compile(func, name, pass_configs=None):
    return tilelang.compile(func.with_attr("global_symbol", name), target="cuda", pass_configs=pass_configs)


@lru_cache(maxsize=None)
def gemm_ws_kernel(M: int, N: int, K: int, cfg: WsConfig, epilogue: str = "store"):
    BM, BN, BK, S, CL = cfg.BM, cfg.BN, cfg.BK, cfg.stages, cfg.CL
    assert BM == 128 and M % BM == 0
    SK = cfg.split_k
    assert K % (SK * BK) == 0
    KS = K // SK                      # K range of one split
    KT = KS // BK                     # k steps per tile
    MT = M // BM
    NTN = (N + BN - 1) // BN
    MCW = cfg.mc == "W" and CL > 1
    if MCW:
        assert MT % CL == 0
        MG, NG = MT // CL, NTN        # cluster = CL M tiles of one N tile
    else:
        MG, NG = MT, (NTN + CL - 1) // CL   # cluster = CL N tiles of one M tile
    CSK = cfg.csk
    UNITS = MG * NG * (1 if CSK else SK)   # work units (one per cluster)
    CD = 2 if CSK else CL             # cluster size
    PERSIST = cfg.grid > 0
    if PERSIST:
        assert cfg.grid % CD == 0
        NCLUS = min(cfg.grid // CD, UNITS)
    else:
        NCLUS = UNITS
    GRID = NCLUS * CD
    TWO = epilogue == "swiglu2"
    RES = epilogue == "residual"
    assert not CSK or (SK == 2 and RES and (cfg.io or cfg.direct) and not PERSIST and CL == 1)
    PDL = cfg.pdl
    # aggressive merge lets the staging tile alias the ring: only safe when every ring slot is
    # consumed before the epilogue writes it (one tile per CTA) and nothing is prefetched into it
    assert not cfg.amerge or (not PERSIST and not cfg.io and not cfg.csk)
    assert not PDL or (not MCW and not cfg.wbox and (cfg.io or not RES))
    PF = min(S, KT)                   # ring slots whose W is prefetched before griddepcontrol.wait
    ODT = F32 if epilogue == "splitk" else BF16
    swz = tilelang.layout.make_swizzled_layout
    MASK = (1 << CL) - 1
    DIRECT = cfg.direct
    IO = cfg.io
    PK = cfg.pk
    assert not (IO and DIRECT)
    WBOX = [(b, min(b + cfg.wbox, BN)) for b in range(0, BN, cfg.wbox)] if cfg.wbox else None
    # 128B-swizzled staging tile only when its rows are whole 64-column panels
    SWZ_OUT = (BN * (4 if ODT == F32 else 2)) % 128 == 0 and BN % 64 == 0

    def tile_coords(u, rank):
        if CSK:
            return u % MT, u // MT, rank
        # unit order: M group fastest, then N group, then split
        mg = u % MG
        r = u // MG
        ng = r % NG
        z = r // NG
        if MCW:
            return mg * CL + rank, ng, z
        return mg, ng * CL + rank, z

    def n_units(cid):
        return T.ceildiv(UNITS - cid, NCLUS) if PERSIST else 1

    @T.macro
    def load_w(n, kk, s, W, Wu, Ws, Us, full):
        T.tma_copy(W[n * BN:(n + 1) * BN, kk:kk + BK], Ws[s, :, :], barrier=full[s], leader_scope_threads=32)
        if TWO:
            T.tma_copy(Wu[n * BN:(n + 1) * BN, kk:kk + BK], Us[s, :, :], barrier=full[s], leader_scope_threads=32)

    @T.macro
    def produce_tile(it, u, rank, A, W, Wu, As, Ws, Us, full, empty):
        m, n, z = tile_coords(u, rank)
        for k in T.serial(KT):
            g = it * KT + k
            s = g % S
            T.mbarrier_wait_parity(empty[s], ((g // S) % 2) ^ 1)
            kk = z * KS + k * BK
            if CL > 1 and not MCW:
                T.tma_copy(A[m * BM:(m + 1) * BM, kk:kk + BK], As[s, :, :], barrier=full[s],
                           cluster_mask=MASK, leader_scope_threads=32)
            elif cfg.debug not in ("noa", "wonly"):
                T.tma_copy(A[m * BM:(m + 1) * BM, kk:kk + BK], As[s, :, :], barrier=full[s],
                           leader_scope_threads=32)
            if MCW:
                T.tma_copy(W[n * BN:(n + 1) * BN, kk:kk + BK], Ws[s, :, :], barrier=full[s],
                           cluster_mask=MASK, leader_scope_threads=32)
                if TWO:
                    T.tma_copy(Wu[n * BN:(n + 1) * BN, kk:kk + BK], Us[s, :, :], barrier=full[s],
                               cluster_mask=MASK, leader_scope_threads=32)
            elif WBOX:
                for b0, b1 in WBOX:
                    T.tma_copy(W[n * BN + b0:n * BN + b1, kk:kk + BK], Ws[s, b0:b1, :], barrier=full[s],
                               leader_scope_threads=32)
                    if TWO:
                        T.tma_copy(Wu[n * BN + b0:n * BN + b1, kk:kk + BK], Us[s, b0:b1, :], barrier=full[s],
                                   leader_scope_threads=32)
            elif PDL:
                if g >= PF:
                    load_w(n, kk, s, W, Wu, Ws, Us, full)
            else:
                if cfg.debug != "now":
                    T.tma_copy(W[n * BN:(n + 1) * BN, kk:kk + BK], Ws[s, :, :], barrier=full[s],
                               leader_scope_threads=32)
                if TWO:
                    T.tma_copy(Wu[n * BN:(n + 1) * BN, kk:kk + BK], Us[s, :, :], barrier=full[s],
                               leader_scope_threads=32)
            T.mbarrier_arrive(full[s])

    @T.macro
    def producer(bid, A, W, Wu, As, Ws, Us, full, empty):
        cid = bid // CD
        rank = bid % CD
        if PDL:
            # weights do not depend on the previous kernel: fill the ring's W part first
            m0, n0, z0 = tile_coords(cid, rank)
            for k in T.serial(PF):
                load_w(n0, z0 * KS + k * BK, k, W, Wu, Ws, Us, full)
            T.pdl_sync()
        if PERSIST:
            for it in T.serial(n_units(cid)):
                produce_tile(it, cid + it * NCLUS, rank, A, W, Wu, As, Ws, Us, full, empty)
        else:
            produce_tile(0, cid, rank, A, W, Wu, As, Ws, Us, full, empty)

    @T.macro
    def release(empty, s, lane):
        if CL > 1:
            if lane < CL:
                T.mbarrier_arrive(empty[s], lane)
        else:
            if lane == 0:
                T.mbarrier_arrive(empty[s])

    @T.macro
    def mainloop(it, As, Ws, Us, Cl, Cu, full, empty, lane):
        T.clear(Cl)
        if TWO:
            T.clear(Cu)
        for k in T.serial(KT):
            g = it * KT + k
            s = g % S
            T.mbarrier_wait_parity(full[s], (g // S) % 2)
            if cfg.debug not in ("nomma", "wonly"):
                T.wgmma_gemm(As[s, :, :], Ws[s, :, :], Cl, transpose_B=True, policy=_POLICY)
                if TWO:
                    T.wgmma_gemm(As[s, :, :], Us[s, :, :], Cu, transpose_B=True, policy=_POLICY)
            if cfg.wait1:
                T.wait_wgmma(1)
                if k > 0:
                    release(empty, (g - 1) % S, lane)
            else:
                T.wait_wgmma(0)
                release(empty, s, lane)
        if cfg.wait1:
            T.wait_wgmma(0)
            release(empty, (it * KT + KT - 1) % S, lane)

    @T.macro
    def swiglu_packed(Cl, Cu, Ob, Gb, Ub):
        # same operations and roundings as below, every f32 -> bf16 cast a packed T.copy
        T.copy(Cl, Gb)
        T.copy(Cu, Ub)
        for i, j in T.Parallel(BM, BN):
            g = T.cast(Gb[i, j], F32)
            Cl[i, j] = g * T.call_extern(F32, "gws_rcp_approx", T.float32(1.0) + T.exp(-g))
        T.copy(Cl, Gb)
        for i, j in T.Parallel(BM, BN):
            Cl[i, j] = T.cast(Gb[i, j], F32) * T.cast(Ub[i, j], F32)
        T.copy(Cl, Ob)

    @T.macro
    def residual_packed(Cl, Ob, Rl):
        T.copy(Cl, Ob)
        for i, j in T.Parallel(BM, BN):
            Cl[i, j] = T.cast(Rl[i, j], F32) + T.cast(Ob[i, j], F32)
        T.copy(Cl, Ob)

    @T.macro
    def epilogue_math(m, n, Cl, Cu, Ob, R, Rl, Gb, Ub):
        if TWO and PK:
            swiglu_packed(Cl, Cu, Ob, Gb, Ub)
        elif RES and PK:
            T.copy(R[m * BM:(m + 1) * BM, n * BN:(n + 1) * BN], Rl)
            residual_packed(Cl, Ob, Rl)
        elif TWO:
            for i, j in T.Parallel(BM, BN):
                g = T.cast(T.cast(Cl[i, j], BF16), F32)
                u = T.cast(T.cast(Cu[i, j], BF16), F32)
                sg = T.cast(g * T.fast_rcp(T.float32(1.0) + T.exp(-g)), BF16)
                Ob[i, j] = T.cast(T.cast(sg, F32) * u, BF16)
        elif RES:
            T.copy(R[m * BM:(m + 1) * BM, n * BN:(n + 1) * BN], Rl)
            for i, j in T.Parallel(BM, BN):
                Ob[i, j] = T.cast(T.cast(Rl[i, j], F32) + T.cast(T.cast(Cl[i, j], BF16), F32), BF16)
        else:
            T.copy(Cl, Ob)

    @T.macro
    def store_tile(m, n, z, Ob, Os, C):
        if PERSIST:
            T.tma_store_wait(0)
        # both consumer warpgroups past their last wgmma before the staging tile is written:
        # with amerge it aliases the A ring, which the other warpgroup's last wgmma may still read
        T.sync_threads(1, 256)
        T.copy(Ob, Os)
        T.fence_proxy_async()
        T.sync_threads(1, 256)
        if SK > 1 and not CSK:
            T.tma_copy(Os, C[z, m * BM:(m + 1) * BM, n * BN:(n + 1) * BN])
        else:
            T.tma_copy(Os, C[m * BM:(m + 1) * BM, n * BN:(n + 1) * BN])

    @T.macro
    def io_tile(it, nt, cid, rank, R, C, Os, rfull, ofull, oempty):
        m, n, z = tile_coords(cid + it * NCLUS, rank)
        T.mbarrier_wait_parity(ofull[0], it % 2)
        if SK > 1 and not CSK:
            T.tma_copy(Os, C[z, m * BM:(m + 1) * BM, n * BN:(n + 1) * BN], leader_scope_threads=32)
        else:
            T.tma_copy(Os, C[m * BM:(m + 1) * BM, n * BN:(n + 1) * BN], leader_scope_threads=32)
        T.tma_store_wait(0)
        if RES:
            if it + 1 < nt:
                m1, n1, z1 = tile_coords(cid + (it + 1) * NCLUS, rank)
                T.tma_copy(R[m1 * BM:(m1 + 1) * BM, n1 * BN:(n1 + 1) * BN], Os, barrier=rfull[0],
                           leader_scope_threads=32)
                T.mbarrier_arrive(rfull[0])
        else:
            T.mbarrier_arrive(oempty[0])

    @T.macro
    def io_warp_body(bid, R, C, Os, rfull, ofull, oempty):
        cid = bid // CD
        rank = bid % CD
        if PDL:
            T.pdl_sync()
        if RES:
            m0, n0, z0 = tile_coords(cid, rank)
            T.tma_copy(R[m0 * BM:(m0 + 1) * BM, n0 * BN:(n0 + 1) * BN], Os, barrier=rfull[0],
                       leader_scope_threads=32)
            T.mbarrier_arrive(rfull[0])
        if PERSIST:
            nt = n_units(cid)
            for it in T.serial(nt):
                io_tile(it, nt, cid, rank, R, C, Os, rfull, ofull, oempty)
        else:
            io_tile(0, 1, cid, rank, R, C, Os, rfull, ofull, oempty)

    @T.macro
    def io_warp(bid, R, C, Os, rfull, ofull, oempty):
        if CSK:
            if bid % 2 == 0:
                io_warp_body(bid, R, C, Os, rfull, ofull, oempty)
        else:
            io_warp_body(bid, R, C, Os, rfull, ofull, oempty)

    @T.macro
    def io_epilogue(it, Cl, Cu, Ob, Os, Rl, Gb, Ub, rfull, ofull, oempty):
        if TWO and PK:
            swiglu_packed(Cl, Cu, Ob, Gb, Ub)
        elif RES and PK:
            T.mbarrier_wait_parity(rfull[0], it % 2)
            T.copy(Os, Rl)
            residual_packed(Cl, Ob, Rl)
        elif TWO:
            for i, j in T.Parallel(BM, BN):
                g = T.cast(T.cast(Cl[i, j], BF16), F32)
                u = T.cast(T.cast(Cu[i, j], BF16), F32)
                sg = T.cast(g * T.fast_rcp(T.float32(1.0) + T.exp(-g)), BF16)
                Ob[i, j] = T.cast(T.cast(sg, F32) * u, BF16)
        elif RES:
            T.mbarrier_wait_parity(rfull[0], it % 2)
            T.copy(Os, Rl)
            for i, j in T.Parallel(BM, BN):
                Ob[i, j] = T.cast(T.cast(Rl[i, j], F32) + T.cast(T.cast(Cl[i, j], BF16), F32), BF16)
        else:
            T.copy(Cl, Ob)
        if not RES:
            T.mbarrier_wait_parity(oempty[0], (it % 2) ^ 1)
        T.copy(Ob, Os)
        T.fence_proxy_async()
        # one arrive per warp (8 in total): 256 arrives on one mbarrier serialise
        T.sync_warp()
        if T.get_thread_binding() % 32 == 0:
            T.mbarrier_arrive(ofull[0])

    @T.macro
    def consume_tile(it, u, rank, As, Ws, Us, Cl, Cu, Ob, Os, full, empty, lane, R, Rl, C, rfull, ofull, oempty, Gb, Ub,
                     Pr, Pl, pfull):
        m, n, z = tile_coords(u, rank)
        mainloop(it, As, Ws, Us, Cl, Cu, full, empty, lane)
        if CSK:
            if rank == 1:
                T.copy(Cl, Pr)
                T.fence_proxy_async()
                T.sync_threads(1, 256)
                T.copy_cluster(Pr, Pr, dst_block=0, remote_barrier=pfull[0])
            else:
                T.mbarrier_wait_parity(pfull[0], 0)
                T.copy(Pr, Pl)
                # the reduce kernel's order: s = 0; s += P[0]; s += P[1]
                for i, j in T.Parallel(BM, BN):
                    Cl[i, j] = (T.float32(0) + Cl[i, j]) + Pl[i, j]
                if IO:
                    io_epilogue(it, Cl, Cu, Ob, Os, Rl, Gb, Ub, rfull, ofull, oempty)
                else:
                    epilogue_math(m, n, Cl, Cu, Ob, R, Rl, Gb, Ub)
                    T.copy(Ob, C[m * BM:(m + 1) * BM, n * BN:(n + 1) * BN])
        elif IO:
            io_epilogue(it, Cl, Cu, Ob, Os, Rl, Gb, Ub, rfull, ofull, oempty)
        elif cfg.debug == "noepi":
            if it == 0:
                epilogue_math(m, n, Cl, Cu, Ob, R, Rl, Gb, Ub)
                store_tile(m, n, z, Ob, Os, C)
        elif DIRECT:
            epilogue_math(m, n, Cl, Cu, Ob, R, Rl, Gb, Ub)
            if SK > 1 and not CSK:
                T.copy(Ob, C[z, m * BM:(m + 1) * BM, n * BN:(n + 1) * BN])
            else:
                T.copy(Ob, C[m * BM:(m + 1) * BM, n * BN:(n + 1) * BN])
        else:
            epilogue_math(m, n, Cl, Cu, Ob, R, Rl, Gb, Ub)
            store_tile(m, n, z, Ob, Os, C)

    @T.macro
    def consumer(bid, As, Ws, Us, Cl, Cu, Ob, Os, full, empty, R, Rl, C, rfull, ofull, oempty, Gb, Ub, Pr, Pl, pfull):
        T.inc_max_nreg(232)
        tx = T.get_thread_binding()
        lane = tx % 32
        cid = bid // CD
        rank = bid % CD
        if PERSIST:
            for it in T.serial(n_units(cid)):
                consume_tile(it, cid + it * NCLUS, rank, As, Ws, Us, Cl, Cu, Ob, Os, full, empty, lane, R, Rl, C,
                             rfull, ofull, oempty, Gb, Ub, Pr, Pl, pfull)
        else:
            consume_tile(0, cid, rank, As, Ws, Us, Cl, Cu, Ob, Os, full, empty, lane, R, Rl, C, rfull, ofull, oempty,
                         Gb, Ub, Pr, Pl, pfull)
        if not DIRECT and not IO:
            T.tma_store_wait(0)

    @T.macro
    def body(bid, A, W, Wu, R, C):
        As = T.alloc_shared((S, BM, BK), BF16)
        Ws = T.alloc_shared((S, BN, BK), BF16)
        Us = T.alloc_shared((S, BN, BK), BF16) if TWO else Ws
        Os = None if DIRECT else T.alloc_shared((BM, BN), ODT)
        Cl = T.alloc_fragment((BM, BN), F32)
        Cu = T.alloc_fragment((BM, BN), F32) if TWO else Cl
        Ob = T.alloc_fragment((BM, BN), ODT)
        Rl = T.alloc_fragment((BM, BN), BF16) if RES else None
        Gb = T.alloc_fragment((BM, BN), BF16) if (TWO and PK) else None
        Ub = T.alloc_fragment((BM, BN), BF16) if (TWO and PK) else None
        Pr = T.alloc_shared((BM, BN), F32) if CSK else None
        Pl = T.alloc_fragment((BM, BN), F32) if CSK else None
        pfull = T.alloc_cluster_barrier([1]) if CSK else None
        full = T.alloc_barrier([32] * S)
        if CL > 1:
            empty = T.alloc_cluster_barrier([8 * CL] * S)
        else:
            empty = T.alloc_barrier([8] * S)
        lay = {As: swz(As), Ws: swz(Ws)}
        if TWO:
            lay[Us] = swz(Us)
        if SWZ_OUT and not DIRECT:
            lay[Os] = swz(Os)
        rfull = T.alloc_barrier([32]) if IO else None
        ofull = T.alloc_barrier([8]) if IO else None
        oempty = T.alloc_barrier([32]) if IO else None
        T.annotate_layout(lay)
        if PDL:
            T.pdl_trigger()
        tx = T.get_thread_binding()
        if tx < 128:
            with T.ws(0):
                T.dec_max_nreg(40)
                if tx < 32:
                    producer(bid, A, W, Wu, As, Ws, Us, full, empty)
                elif tx < 64:
                    if IO:
                        io_warp(bid, R, C, Os, rfull, ofull, oempty)
        else:
            with T.ws(1, 2):
                consumer(bid, As, Ws, Us, Cl, Cu, Ob, Os, full, empty, R, Rl, C, rfull, ofull, oempty, Gb, Ub, Pr, Pl,
                         pfull)
        if CD > 1:
            T.cluster_arrive()
            T.cluster_wait()

    if epilogue == "store":

        @T.prim_func
        def main(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16), C: T.Tensor((M, N), BF16)):
            with T.Kernel(GRID, threads=384, cluster_dims=(CD, 1, 1), prelude=_PRELUDE) as bid:
                body(bid, A, W, W, None, C)

    elif epilogue == "splitk":

        @T.prim_func
        def main(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16), C: T.Tensor((SK, M, N), F32)):
            with T.Kernel(GRID, threads=384, cluster_dims=(CD, 1, 1), prelude=_PRELUDE) as bid:
                body(bid, A, W, W, None, C)

    elif epilogue == "residual":

        @T.prim_func
        def main(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16), R: T.Tensor((M, N), BF16),
                 C: T.Tensor((M, N), BF16)):
            with T.Kernel(GRID, threads=384, cluster_dims=(CD, 1, 1), prelude=_PRELUDE) as bid:
                body(bid, A, W, W, R, C)

    elif epilogue == "swiglu2":

        @T.prim_func
        def main(A: T.Tensor((M, K), BF16), Wg: T.Tensor((N, K), BF16), Wu: T.Tensor((N, K), BF16),
                 O: T.Tensor((M, N), BF16)):
            with T.Kernel(GRID, threads=384, cluster_dims=(CD, 1, 1), prelude=_PRELUDE) as bid:
                body(bid, A, Wg, Wu, None, O)

    else:
        raise ValueError(epilogue)
    pc = {"tl.enable_aggressive_shared_memory_merge": True} if cfg.amerge else None
    return _compile(main, f"gemmws_{epilogue}_{M}x{N}x{K}_{cfg.tag()}", pc)


# Per-length choice (GEMM_WS_NOTES.md, reports/gemm_ws.json): (WsConfig, epilogue) per linear,
# used where the final paired timing was >= 1 % faster than today's kernel; None = keep
# runtime/kernels/gemm.py gemm_kernel with the plan v3 config.
W_ = WsConfig
WS_CONFIGS = {
    128: {
        "gdn_in": None,
        "att_in": None,
        "gate_up": None,
        # replaces gemm "splitk" (BN 80, split 2) + splitk_reduce_kernel; call k(A, W, R, C)
        "o_proj": (W_(BN=80, stages=6, split_k=2, io=True, pk=True, csk=True), "residual"),
        "down": (W_(BN=80, stages=6, split_k=2, io=True, pk=True, csk=True), "residual"),
    },
    256: {
        "gdn_in": None,
        "att_in": None,
        "gate_up": (W_(BN=144, stages=4, pk=True, amerge=True), "swiglu2"),
        "o_proj": None,
        "down": None,
    },
    512: {
        "gdn_in": (W_(BN=256, stages=3, grid=NSM), "store"),
        "att_in": None,
        "gate_up": (W_(BN=176, stages=3, grid=NSM, pk=True), "swiglu2"),
        "o_proj": (W_(BN=160, stages=4, io=True, pk=True), "residual"),
        "down": None,
    },
    1024: {
        "gdn_in": (W_(BN=256, stages=3, grid=NSM), "store"),
        "att_in": (W_(BN=224, stages=3, grid=NSM), "store"),
        "gate_up": (W_(BN=176, stages=3, grid=NSM, pk=True), "swiglu2"),
        "o_proj": (W_(BN=160, stages=4, grid=NSM, pk=True), "residual"),
        "down": (W_(BN=160, stages=4, grid=NSM, pk=True), "residual"),
    },
}
