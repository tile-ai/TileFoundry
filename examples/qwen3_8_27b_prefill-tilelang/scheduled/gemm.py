"""Scheduled HIR for the prefill GEMMs: model.py's projection functions with
their placement and instruction choices stated (`tf.schedule`).

One Module per (model.py function, sequence length). Its entry @func has the
model.py function's name and signature, so `tilefoundry check` holds it to
model.py's reference; the @funcs it calls are the kernels kernels/gemm.py
implements, and `tilefoundry schedule finalize` lowers each to the TIR in
tir/:

  pack_*     weights only (ConstTensor in, ConstTensor out): the packed weight
             one GEMM reads. The runtime runs it once and holds the result.
  *_main     132 persistent CTAs x 384 threads:
               threads[0:2, :]  two consumer warpgroups, 64 rows each, one
                                m64n256k16 WGMMA per 16 of K (A and B K-major,
                                SW128 smem), accumulating in registers;
               threads[2, :32]  the producer: 64-wide K slabs of A and B by
                                TMA through a 3-deep ring (buffers=3).
             Work of CTA c, in order:
               1. tail pieces (only when the plan has a tail): the last R
                  tiles in raster order -- whole bands B0.. of the last M
                  group -- are cut along K into S pieces; piece p of tail
                  tile tt is CTA tt * S + p. Its f32 partial goes to ws.
               2. whole tiles: tile t = (g * NB + bn) * GROUP + mi goes to CTA
                  t mod 132 (GROUP M-tiles walk down one band first, so a
                  wave's weight columns stay in L2).
             Epilogue on the consumers from bf16(acc), staged in SW128 smem,
             stored by TMA from warp 0 of the consumers (residual: the hidden
             tile loaded by TMA into the same smem first; swiglu: gate and up
             read back on a 16 x 16 grid, the product over the gate half;
             qkv: q / k heads RMS-normed and rotated on a 16 x 16 grid
             through the staged tile, under guard loops on the band).
  *_tail     one CTA per (tail tile, RT-row chunk), 384 threads of which 256
             (RT rows x TC lanes) participate: the S partials summed in piece
             order (so the sum does not depend on timing), then the same
             epilogue; lane c of a row holds columns (k * TC + c) * 4 .. + 3
             (16-byte runs interleaved across the lanes). It reads pieces
             other CTAs wrote, so it is a Module of its own, `<Name>Tail<n>`
             on its own grid (NBT x GROUP x CY CTAs): the host invokes it
             after `<Name><n>`, whose entry returns (out, ws). The launch
             boundary is the grid-wide barrier the ws handover needs.

The tile plans (kernels/gemm.py `plan`, evidence/STEP4.md):
             1024                4096       8192
  in_proj    520 tiles           2080       4160
  gate_up    1088 = 1056 + 32x4  4352       8704
  down       160 = 128 + 32x4    640        1280
  out/o_proj 160 = 128 + 32x4    640        1280
  qkv        448 = 392 + 56x2    1792       3584 = 3552 + 32x4
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
CTAS, BM, BN, BK, STAGES = 132, 128, 256, 64, 3
H, I = 5120, 17408
HQ, HKV, HD, ROT, HALF = 24, 4, 256, 64, 32   # attention heads, head dim, rotary dims
QKV_W = HQ * 2 * HD + 2 * HKV * HD          # 14336
NQG, NKH = HQ * 2, HKV                       # q|gate column blocks, k heads
EPS = 1e-6


def _plan(M: int, N: int, K: int):
    """kernels/gemm.py `plan`, restated: GROUP = 16 M-tiles where M has that
    many (4096, 8192), else 8; tail = whole bands of the last group."""
    mb, nb, kb = M // BM, N // BN, K // BK
    GROUP = 16 if mb % 16 == 0 else (8 if mb % 8 == 0 else mb)
    tiles = mb * nb
    rem = -(-(tiles % CTAS) // GROUP) * GROUP
    s = 1
    if rem:
        s = max(d for d in range(1, min(CTAS // rem, 8) + 1) if kb % d == 0)
    if s < 2 or rem >= tiles:
        rem, s = 0, 1
    cy = 1
    while rem and rem * cy * 2 <= CTAS and cy < 8:
        cy *= 2
    return dict(MB=mb, NB=nb, KB=kb, REM=rem, S=s, KPP=kb // s, NBT=rem // GROUP, GROUP=GROUP,
                B0=nb - rem // GROUP, G0=mb // GROUP - 1, UNITS=max(rem * s, 1),
                CY=cy, RT=BM // cy)  # tail: CY row chunks of RT rows per tile


def residual(fname: str, M: int, K: int):
    """`out = hidden + x @ w^T` (MLP.down, GatedDeltaNet.out_proj, Attention.o_proj)."""
    p = _plan(M, H, K)
    NB, B0, G0, NBT, S, KPP, UNITS = p["NB"], p["B0"], p["G0"], p["NBT"], p["S"], p["KPP"], p["UNITS"]
    GROUP = p["GROUP"]
    CY, RT = p["CY"], p["RT"]
    HAS_TAIL = p["REM"] > 0
    TC = 256 // RT                       # tail: threads per row
    GFULL = G0 if p["REM"] else G0 + 1   # M groups whose every band is whole tiles
    BLAST = B0 if p["REM"] else 0        # whole bands of the last group, when it has a tail

    @module(entry="entry", target=_H200, topologies=(Topology("cta", CTAS), Topology("thread", 384)))
    class Residual:

        if HAS_TAIL:
            @func
            def pieces(x: Tensor[(1, M, K), "bf16"], w: ConstTensor[(H, K), "bf16"]):
                with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                    with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                        wgmma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                                  a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                        a = tf.reshape(x, (M, K))
                        ws = tf.zeros(Tensor[(UNITS * BM, BN), "f32"])
                        # 1. tail pieces: K blocks [p * KPP, (p + 1) * KPP) of tail tile tt
                        for bb in range(B0, NB):
                            for mi in range(GROUP):
                                for pc in range((cta.persistent - ((bb - B0) * GROUP + mi) * S) % CTAS, S, CTAS):
                                    m = (G0 * GROUP + mi) * BM
                                    n = bb * BN
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                                    for kk in range(KPP):
                                        k = (pc * KPP + kk) * BK
                                        with threads[2, :32] as _loader:
                                            lhs = tf.schedule((a[m:m + BM, k:k + BK],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                            rhs = tf.schedule((w[n:n + BN, k:k + BK],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                        with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                            acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                              op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        ws = tf.insert_slice(ws, acc, ((((bb - B0) * GROUP + mi) * S + pc) * BM, 0))
                        return ws

        @func
        def whole(x: Tensor[(1, M, K), "bf16"], hidden: Tensor[(1, M, H), "bf16"], w: ConstTensor[(H, K), "bf16"]):
            with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                    wgmma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                              a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                    a = tf.reshape(x, (M, K))
                    r = tf.reshape(hidden, (M, H))
                    out = tf.zeros(Tensor[(M, H), "bf16"])
                    # 2. whole tiles: full M groups, then the last group's whole bands
                    for g in range(GFULL):
                        for bn in range(NB):
                            for mi in range((cta.persistent - (g * NB + bn) * GROUP) % CTAS, GROUP, CTAS):
                                m = (g * GROUP + mi) * BM
                                n = bn * BN
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                                for k in tf.tile(K, BK):
                                    with threads[2, :32] as _loader:
                                        lhs = tf.schedule((a[m:m + BM, k],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                        rhs = tf.schedule((w[n:n + BN, k],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                          op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                                with threads[0, :32] as _rload:
                                    res = tf.schedule((r[m:m + BM, n:n + BN],), op=T.copy_async_tensor(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    y = tf.schedule((res,), op=T.copy(rmem_layout=((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)))) + tf.cast(acc, "bf16")
                                    staged = tf.schedule((y,), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                with threads[0, :32] as _storer:
                                    tile = tf.schedule((staged,), op=T.copy_async_tensor())
                                    out = tf.insert_slice(out, tile, (m, n))
                    for bn in range(BLAST):
                        for mi in range((cta.persistent - (G0 * NB + bn) * GROUP) % CTAS, GROUP, CTAS):
                            m = (G0 * GROUP + mi) * BM
                            n = bn * BN
                            with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                            for k in tf.tile(K, BK):
                                with threads[2, :32] as _loader:
                                    lhs = tf.schedule((a[m:m + BM, k],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                    rhs = tf.schedule((w[n:n + BN, k],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                      op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                            with threads[0, :32] as _rload:
                                res = tf.schedule((r[m:m + BM, n:n + BN],), op=T.copy_async_tensor(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                            with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                y = tf.schedule((res,), op=T.copy(rmem_layout=((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)))) + tf.cast(acc, "bf16")
                                staged = tf.schedule((y,), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                            with threads[0, :32] as _storer:
                                tile = tf.schedule((staged,), op=T.copy_async_tensor())
                                out = tf.insert_slice(out, tile, (m, n))
                    return out

        if HAS_TAIL:
            @func
            def entry(x: Tensor[(1, M, K), "bf16"], hidden: Tensor[(1, M, H), "bf16"], w: ConstTensor[(H, K), "bf16"]):
                ws = pieces(x, w)  # noqa: F821
                return whole(x, hidden, w), ws  # noqa: F821

        else:
            @func
            def entry(x: Tensor[(1, M, K), "bf16"], hidden: Tensor[(1, M, H), "bf16"], w: ConstTensor[(H, K), "bf16"]) -> Tensor[(1, M, H), "bf16"]:
                return tf.reshape(whole(x, hidden, w), (1, M, H))  # noqa: F821

    Residual.__doc__ = f"{fname} at seq_len={M}: {p['MB'] * NB} tiles, tail {p['REM']} x {S}"
    if not HAS_TAIL:
        return Residual, None

    @module(entry="entry", target=_H200, topologies=(Topology("cta", NBT * GROUP * CY), Topology("thread", 384)))
    class ResidualTail:
        @func
        def tail(out: Tensor[(M, H), "bf16"], ws: Tensor[(UNITS * BM, BN), "f32"], hidden: Tensor[(1, M, H), "bf16"]):
            with Mesh(("cta",), layout=(NBT, GROUP, CY), names=("tb", "mi", "cy")) as cta:
                with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                    with Mesh(threads[0:2, :], layout=(RT, TC), names=("row", "col")) as th:
                        r = tf.reshape(hidden, (M, H))
                        t = cta.tb * GROUP + cta.mi
                        acc = tf.schedule((ws[t * S * BM + cta.cy * RT:t * S * BM + cta.cy * RT + RT, :],), op=T.copy(rmem_layout=(RT @ th.row, BN // (4 * TC), TC @ th.col, 4)))
                        for j in range(1, S):
                            acc = acc + tf.schedule((ws[(t * S + j) * BM + cta.cy * RT:(t * S + j) * BM + cta.cy * RT + RT, :],), op=T.copy(rmem_layout=(RT @ th.row, BN // (4 * TC), TC @ th.col, 4)))
                        m = (G0 * GROUP + cta.mi) * BM + cta.cy * RT
                        n = (B0 + cta.tb) * BN
                        y = tf.schedule((r[m:m + RT, n:n + BN],), op=T.copy(rmem_layout=(RT @ th.row, BN // (4 * TC), TC @ th.col, 4))) + tf.cast(acc, "bf16")
                        return tf.insert_slice(out, y, (m, n))

        @func
        def entry(out: Tensor[(M, H), "bf16"], ws: Tensor[(UNITS * BM, BN), "f32"], hidden: Tensor[(1, M, H), "bf16"]) -> Tensor[(1, M, H), "bf16"]:
            return tf.reshape(tail(out, ws, hidden), (1, M, H))  # noqa: F821

    ResidualTail.__doc__ = f"{fname} at seq_len={M}, invocation 2 of 2: the split tail ({p['REM']} tiles x {S} pieces)"
    return Residual, ResidualTail


def gate_up(M: int):
    """MLP.gate_up: silu(hn @ w_gate^T) * (hn @ w_up^T), each landing in bf16.

    One m64n256 WGMMA per 16 of K over the packed weight [128 gate | 128 up]
    rows per tile; output tile 128 x 128."""
    K, N, NO = H, 2 * I, I
    p = _plan(M, N, K)
    NB, B0, G0, NBT, S, KPP, UNITS = p["NB"], p["B0"], p["G0"], p["NBT"], p["S"], p["KPP"], p["UNITS"]
    GROUP = p["GROUP"]
    CY, RT = p["CY"], p["RT"]
    HAS_TAIL = p["REM"] > 0
    TC = 256 // RT
    GFULL = G0 if p["REM"] else G0 + 1
    BLAST = B0 if p["REM"] else 0

    @module(entry="entry", target=_H200, topologies=(Topology("cta", CTAS), Topology("thread", 384)))
    class GateUp:
        @func
        def pack(w_gate: ConstTensor[(I, H), "bf16"], w_up: ConstTensor[(I, H), "bf16"]) -> Tensor[(2 * I, H), "bf16"]:
            with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                with Mesh(("thread",), layout=(384,), names=("t",)) as _th:
                    w = tf.zeros(Tensor[(2 * I, H), "bf16"])
                    for j in range(cta.persistent, I // 128, CTAS):
                        w = tf.insert_slice(w, w_gate[j * 128:j * 128 + 128, :], (j * 256, 0))
                        w = tf.insert_slice(w, w_up[j * 128:j * 128 + 128, :], (j * 256 + 128, 0))
                    return w

        if HAS_TAIL:
            @func
            def pieces(x: Tensor[(1, M, K), "bf16"], w: Tensor[(N, K), "bf16"]):
                with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                    with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                        wgmma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                                  a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                        a = tf.reshape(x, (M, K))
                        ws = tf.zeros(Tensor[(UNITS * BM, BN), "f32"])
                        for bb in range(B0, NB):
                            for mi in range(GROUP):
                                for pc in range((cta.persistent - ((bb - B0) * GROUP + mi) * S) % CTAS, S, CTAS):
                                    m = (G0 * GROUP + mi) * BM
                                    n = bb * BN
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                                    for kk in range(KPP):
                                        k = (pc * KPP + kk) * BK
                                        with threads[2, :32] as _loader:
                                            lhs = tf.schedule((a[m:m + BM, k:k + BK],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                            rhs = tf.schedule((w[n:n + BN, k:k + BK],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                        with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                            acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                              op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        ws = tf.insert_slice(ws, acc, ((((bb - B0) * GROUP + mi) * S + pc) * BM, 0))
                        return ws

        @func
        def whole(x: Tensor[(1, M, K), "bf16"], w: Tensor[(N, K), "bf16"]):
            with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                    wgmma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                              a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                    a = tf.reshape(x, (M, K))
                    out = tf.zeros(Tensor[(M, NO), "bf16"])
                    for g in range(GFULL):
                        for bn in range(NB):
                            for mi in range((cta.persistent - (g * NB + bn) * GROUP) % CTAS, GROUP, CTAS):
                                m = (g * GROUP + mi) * BM
                                n = bn * BN
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                                for k in tf.tile(K, BK):
                                    with threads[2, :32] as _loader:
                                        lhs = tf.schedule((a[m:m + BM, k],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                        rhs = tf.schedule((w[n:n + BN, k],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                          op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    cs = tf.schedule((tf.cast(acc, "bf16"),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                with Mesh(threads[0:2, :], layout=(16, 16), names=("r", "c")) as e:
                                    g32 = tf.cast(tf.schedule((cs[:, 0:128],), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 8))), "f32")
                                    u = tf.schedule((cs[:, 128:256],), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 8)))
                                    act = tf.cast(g32 / (1.0 + tf.exp(-g32)), "bf16") * u
                                    staged = tf.schedule((act,), op=T.copy(smem_layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                with threads[0, :32] as _storer:
                                    tile = tf.schedule((staged,), op=T.copy_async_tensor())
                                    out = tf.insert_slice(out, tile, (m, bn * 128))
                    for bn in range(BLAST):
                        for mi in range((cta.persistent - (G0 * NB + bn) * GROUP) % CTAS, GROUP, CTAS):
                            m = (G0 * GROUP + mi) * BM
                            n = bn * BN
                            with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                            for k in tf.tile(K, BK):
                                with threads[2, :32] as _loader:
                                    lhs = tf.schedule((a[m:m + BM, k],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                    rhs = tf.schedule((w[n:n + BN, k],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                      op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                            with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                cs = tf.schedule((tf.cast(acc, "bf16"),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                            with Mesh(threads[0:2, :], layout=(16, 16), names=("r", "c")) as e:
                                g32 = tf.cast(tf.schedule((cs[:, 0:128],), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 8))), "f32")
                                u = tf.schedule((cs[:, 128:256],), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 8)))
                                act = tf.cast(g32 / (1.0 + tf.exp(-g32)), "bf16") * u
                                staged = tf.schedule((act,), op=T.copy(smem_layout=Layout((128, (2, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                            with threads[0, :32] as _storer:
                                tile = tf.schedule((staged,), op=T.copy_async_tensor())
                                out = tf.insert_slice(out, tile, (m, bn * 128))
                    return out

        if HAS_TAIL:
            @func
            def entry(hn: Tensor[(1, M, H), "bf16"], w_gate: ConstTensor[(I, H), "bf16"], w_up: ConstTensor[(I, H), "bf16"]):
                w = pack(w_gate, w_up)  # noqa: F821
                ws = pieces(hn, w)  # noqa: F821
                return whole(hn, w), ws  # noqa: F821
        else:
            @func
            def entry(hn: Tensor[(1, M, H), "bf16"], w_gate: ConstTensor[(I, H), "bf16"], w_up: ConstTensor[(I, H), "bf16"]) -> Tensor[(1, M, I), "bf16"]:
                return tf.reshape(whole(hn, pack(w_gate, w_up)), (1, M, I))  # noqa: F821

    GateUp.__doc__ = f"gate_up at seq_len={M}: {p['MB'] * NB} tiles, tail {p['REM']} x {S}"
    if not HAS_TAIL:
        return GateUp, None

    @module(entry="entry", target=_H200, topologies=(Topology("cta", NBT * GROUP * CY), Topology("thread", 384)))
    class GateUpTail:
        @func
        def tail(out: Tensor[(M, NO), "bf16"], ws: Tensor[(UNITS * BM, BN), "f32"]):
            with Mesh(("cta",), layout=(NBT, GROUP, CY), names=("tb", "mi", "cy")) as cta:
                with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                    with Mesh(threads[0:2, :], layout=(RT, TC), names=("row", "col")) as th:
                        t = cta.tb * GROUP + cta.mi
                        r0 = t * S * BM + cta.cy * RT
                        acc_g = tf.schedule((ws[r0:r0 + RT, 0:128],), op=T.copy(rmem_layout=(RT @ th.row, 128 // (4 * TC), TC @ th.col, 4)))
                        acc_u = tf.schedule((ws[r0:r0 + RT, 128:256],), op=T.copy(rmem_layout=(RT @ th.row, 128 // (4 * TC), TC @ th.col, 4)))
                        for j in range(1, S):
                            acc_g = acc_g + tf.schedule((ws[r0 + j * BM:r0 + j * BM + RT, 0:128],), op=T.copy(rmem_layout=(RT @ th.row, 128 // (4 * TC), TC @ th.col, 4)))
                            acc_u = acc_u + tf.schedule((ws[r0 + j * BM:r0 + j * BM + RT, 128:256],), op=T.copy(rmem_layout=(RT @ th.row, 128 // (4 * TC), TC @ th.col, 4)))
                        g32 = tf.cast(tf.cast(acc_g, "bf16"), "f32")
                        act = tf.cast(g32 / (1.0 + tf.exp(-g32)), "bf16") * tf.cast(acc_u, "bf16")
                        m = (G0 * GROUP + cta.mi) * BM + cta.cy * RT
                        return tf.insert_slice(out, act, (m, (B0 + cta.tb) * 128))

        @func
        def entry(out: Tensor[(M, NO), "bf16"], ws: Tensor[(UNITS * BM, BN), "f32"]) -> Tensor[(1, M, I), "bf16"]:
            return tf.reshape(tail(out, ws), (1, M, I))  # noqa: F821

    GateUpTail.__doc__ = f"gate_up at seq_len={M}, invocation 2 of 2: the split tail ({p['REM']} tiles x {S} pieces)"
    return GateUp, GateUpTail


def in_proj(M: int):
    """GatedDeltaNet.in_proj: one GEMM over the packed [qkv | z | b | a | 0]
    weight (16640 rows = 65 whole tiles); no tail at 1024 / 4096 / 8192."""
    K, N = H, 16640
    p = _plan(M, N, K)
    NB, G0, GROUP = p["NB"], p["G0"], p["GROUP"]
    assert p["REM"] == 0

    @module(entry="entry", target=_H200, topologies=(Topology("cta", CTAS), Topology("thread", 384)))
    class InProj:
        @func
        def pack(w_qkv: ConstTensor[(10240, H), "bf16"], w_z: ConstTensor[(6144, H), "bf16"],
                 w_b: ConstTensor[(48, H), "bf16"], w_a: ConstTensor[(48, H), "bf16"]) -> Tensor[(N, H), "bf16"]:
            with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                with Mesh(("thread",), layout=(384,), names=("t",)) as _th:
                    w = tf.zeros(Tensor[(N, H), "bf16"])
                    for j in range(cta.persistent, 80, CTAS):
                        w = tf.insert_slice(w, w_qkv[j * 128:j * 128 + 128, :], (j * 128, 0))
                    for j in range(cta.persistent, 48, CTAS):
                        w = tf.insert_slice(w, w_z[j * 128:j * 128 + 128, :], (10240 + j * 128, 0))
                    for j in range(cta.persistent, 1, CTAS):
                        w = tf.insert_slice(w, w_b, (16384, 0))
                        w = tf.insert_slice(w, w_a, (16432, 0))
                    return w

        @func
        def whole(x: Tensor[(1, M, K), "bf16"], w: Tensor[(N, K), "bf16"]):
            with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                    wgmma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                              a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                    a = tf.reshape(x, (M, K))
                    out = tf.zeros(Tensor[(M, N), "bf16"])
                    for g in range(G0 + 1):
                        for bn in range(NB):
                            for mi in range((cta.persistent - (g * NB + bn) * GROUP) % CTAS, GROUP, CTAS):
                                m = (g * GROUP + mi) * BM
                                n = bn * BN
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                                for k in tf.tile(K, BK):
                                    with threads[2, :32] as _loader:
                                        lhs = tf.schedule((a[m:m + BM, k],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                        rhs = tf.schedule((w[n:n + BN, k],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                          op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    staged = tf.schedule((tf.cast(acc, "bf16"),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                with threads[0, :32] as _storer:
                                    tile = tf.schedule((staged,), op=T.copy_async_tensor())
                                    out = tf.insert_slice(out, tile, (m, n))
                    return out

        @func
        def entry(hn: Tensor[(1, M, H), "bf16"], w_qkv: ConstTensor[(10240, H), "bf16"], w_z: ConstTensor[(6144, H), "bf16"],
                  w_b: ConstTensor[(48, H), "bf16"], w_a: ConstTensor[(48, H), "bf16"]):
            o = tf.reshape(whole(hn, pack(w_qkv, w_z, w_b, w_a)), (1, M, N))  # noqa: F821
            return o[:, :, 0:10240], o[:, :, 10240:16384], o[:, :, 16384:16432], o[:, :, 16432:16480]

    InProj.__doc__ = f"in_proj at seq_len={M}: {p['MB'] * NB} tiles"
    return InProj, None


def qkv(M: int):
    """Attention.qkv: one GEMM over the packed [q|gate per head | k | v]
    weight (14336 rows); in the epilogue, q and k heads (a 256-wide tile is
    one head) are RMS-normed and rotated (rotate-half on the leading 64 of
    each head; every product and the sum land in bf16), on a 16 x 16 grid of
    the consumer threads through the staged tile. One loop over the column
    blocks per M group (one TMA ring); the head kind of block bn is chosen by
    guard loops of 0 or 1 iterations (q: bn even below 48; k: 48 <= bn < 52);
    gate and v tiles are stored as they land."""
    K, N = H, QKV_W
    p = _plan(M, N, K)
    NB, B0, G0, NBT, S, KPP, UNITS = p["NB"], p["B0"], p["G0"], p["NBT"], p["S"], p["KPP"], p["UNITS"]
    GROUP = p["GROUP"]
    CY, RT = p["CY"], p["RT"]
    HAS_TAIL = p["REM"] > 0
    TC = 256 // RT
    GFULL = G0 if p["REM"] else G0 + 1
    GA = min(GFULL, 2)                               # full groups in `whole`; the rest in `whole_b`
    BLAST = B0 if p["REM"] else 0
    assert not HAS_TAIL or B0 >= NQG                 # the tail holds k / v bands only (1024: 49-55, 8192: 54-55)
    TBK0 = max(NQG, B0) - B0                         # first tail band that is a k head ...
    NKT = max(0, min(NQG + NKH, NB) - max(NQG, B0))  # ... and how many

    @module(entry="entry", target=_H200, topologies=(Topology("cta", CTAS), Topology("thread", 384)))
    class QKV:
        @func
        def pack(w_q: ConstTensor[(HQ * 2 * HD, H), "bf16"], w_k: ConstTensor[(HKV * HD, H), "bf16"],
                 w_v: ConstTensor[(HKV * HD, H), "bf16"]) -> Tensor[(N, H), "bf16"]:
            with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                with Mesh(("thread",), layout=(384,), names=("t",)) as _th:
                    w = tf.zeros(Tensor[(N, H), "bf16"])
                    for j in range(cta.persistent, HQ * 2 * HD // 128, CTAS):
                        w = tf.insert_slice(w, w_q[j * 128:j * 128 + 128, :], (j * 128, 0))
                    for j in range(cta.persistent, HKV * HD // 128, CTAS):
                        w = tf.insert_slice(w, w_k[j * 128:j * 128 + 128, :], (HQ * 2 * HD + j * 128, 0))
                        w = tf.insert_slice(w, w_v[j * 128:j * 128 + 128, :], (HQ * 2 * HD + HKV * HD + j * 128, 0))
                    return w

        if HAS_TAIL:
            @func
            def pieces(x: Tensor[(1, M, K), "bf16"], w: Tensor[(N, K), "bf16"]):
                with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                    with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                        wgmma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                                  a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                        a = tf.reshape(x, (M, K))
                        ws = tf.zeros(Tensor[(UNITS * BM, BN), "f32"])
                        for bb in range(B0, NB):
                            for mi in range(GROUP):
                                for pc in range((cta.persistent - ((bb - B0) * GROUP + mi) * S) % CTAS, S, CTAS):
                                    m = (G0 * GROUP + mi) * BM
                                    n = bb * BN
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                                    for kk in range(KPP):
                                        k = (pc * KPP + kk) * BK
                                        with threads[2, :32] as _loader:
                                            lhs = tf.schedule((a[m:m + BM, k:k + BK],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                            rhs = tf.schedule((w[n:n + BN, k:k + BK],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                        with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                            acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                              op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        ws = tf.insert_slice(ws, acc, ((((bb - B0) * GROUP + mi) * S + pc) * BM, 0))
                        return ws

        @func
        def whole(x: Tensor[(1, M, K), "bf16"], w: Tensor[(N, K), "bf16"], gamma_q: ConstTensor[(HD,), "bf16"],
                  gamma_k: ConstTensor[(HD,), "bf16"], cos: Tensor[(M, ROT), "bf16"], sin: Tensor[(M, ROT), "bf16"]):
            with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                    wgmma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                              a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                    a = tf.reshape(x, (M, K))
                    out = tf.zeros(Tensor[(M, N), "bf16"])
                    for g in range(GA):
                        for bn in range(NB):   # the column blocks in order: q|gate per head, k heads, v heads
                            for mi in range((cta.persistent - (g * NB + bn) * GROUP) % CTAS, GROUP, CTAS):
                                m = (g * GROUP + mi) * BM
                                n = bn * BN
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                                for k in tf.tile(K, BK):
                                    with threads[2, :32] as _loader:
                                        lhs = tf.schedule((a[m:m + BM, k],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                        rhs = tf.schedule((w[n:n + BN, k],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                          op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    cs = tf.schedule((tf.cast(acc, "bf16"),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                for _q in range((cta.persistent * 0 + bn) % NB, NQG, 1048576):   # once while bn is a q|gate block
                                    for _h in range((cta.persistent * 0 + bn) % 2, 1, 2):   # once for a q head (bn even); a gate head is stored as it lands
                                        with Mesh(threads[0:2, :], layout=(16, 16), names=("r", "c")) as e:
                                            x32 = tf.cast(tf.schedule((tf.reshape(cs, (16, 8, 16, 16)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 16))), "f32")
                                            sq = tf.schedule((x32 * x32,), op=T.copy(rmem_layout=((16 @ e.r, 8, 16 @ e.c, 16), (0, 16, 0, 1))))
                                            ms = tf.schedule((tf.reduce(sq, (2, 3), True, ReduceKind.MEAN),), op=T.copy(rmem_layout=(16 @ e.r, 8, 1, 1)))
                                            gw = tf.cast(tf.schedule((tf.reshape(gamma_q, (1, 1, 16, 16)),), op=T.copy(rmem_layout=(1, 1, 16 @ e.c, 16))), "f32") + 1.0
                                            nrm = tf.cast(x32 * tf.rsqrt(ms + EPS) * gw, "bf16")
                                            cs = tf.schedule((tf.reshape(nrm, (BM, BN)),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                            lo = tf.cast(tf.schedule((tf.reshape(cs[:, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            hi = tf.cast(tf.schedule((tf.reshape(cs[:, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            cl = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            ch = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            sl = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            sh = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            rlo = tf.cast(tf.cast(tf.cast(lo * cl, "bf16"), "f32") + tf.cast(tf.cast(-hi * sl, "bf16"), "f32"), "bf16")
                                            rhi = tf.cast(tf.cast(tf.cast(hi * ch, "bf16"), "f32") + tf.cast(tf.cast(lo * sh, "bf16"), "f32"), "bf16")
                                            cs = tf.insert_slice(cs, tf.reshape(rlo, (BM, HALF)), (0, 0))
                                            cs = tf.insert_slice(cs, tf.reshape(rhi, (BM, HALF)), (0, HALF))
                                for _k0 in range(NQG - 1, (cta.persistent * 0 + bn) % NB, 1048576):   # once from the first k block ...
                                    for _k1 in range((cta.persistent * 0 + bn) % NB, NQG + NKH, 1048576):   # ... to the last: a k head (v heads are stored as they land)
                                        with Mesh(threads[0:2, :], layout=(16, 16), names=("r", "c")) as e:
                                            x32 = tf.cast(tf.schedule((tf.reshape(cs, (16, 8, 16, 16)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 16))), "f32")
                                            sq = tf.schedule((x32 * x32,), op=T.copy(rmem_layout=((16 @ e.r, 8, 16 @ e.c, 16), (0, 16, 0, 1))))
                                            ms = tf.schedule((tf.reduce(sq, (2, 3), True, ReduceKind.MEAN),), op=T.copy(rmem_layout=(16 @ e.r, 8, 1, 1)))
                                            gw = tf.cast(tf.schedule((tf.reshape(gamma_k, (1, 1, 16, 16)),), op=T.copy(rmem_layout=(1, 1, 16 @ e.c, 16))), "f32") + 1.0
                                            nrm = tf.cast(x32 * tf.rsqrt(ms + EPS) * gw, "bf16")
                                            cs = tf.schedule((tf.reshape(nrm, (BM, BN)),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                            lo = tf.cast(tf.schedule((tf.reshape(cs[:, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            hi = tf.cast(tf.schedule((tf.reshape(cs[:, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            cl = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            ch = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            sl = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            sh = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            rlo = tf.cast(tf.cast(tf.cast(lo * cl, "bf16"), "f32") + tf.cast(tf.cast(-hi * sl, "bf16"), "f32"), "bf16")
                                            rhi = tf.cast(tf.cast(tf.cast(hi * ch, "bf16"), "f32") + tf.cast(tf.cast(lo * sh, "bf16"), "f32"), "bf16")
                                            cs = tf.insert_slice(cs, tf.reshape(rlo, (BM, HALF)), (0, 0))
                                            cs = tf.insert_slice(cs, tf.reshape(rhi, (BM, HALF)), (0, HALF))
                                with threads[0, :32] as _storer:
                                    tile = tf.schedule((cs,), op=T.copy_async_tensor())
                                    out = tf.insert_slice(out, tile, (m, n))
                    return out

        if GFULL > GA:
            @func
            def whole_b(out: Tensor[(M, N), "bf16"], x: Tensor[(1, M, K), "bf16"], w: Tensor[(N, K), "bf16"],
                        gamma_q: ConstTensor[(HD,), "bf16"], gamma_k: ConstTensor[(HD,), "bf16"], cos: Tensor[(M, ROT), "bf16"], sin: Tensor[(M, ROT), "bf16"]):
                """Full M groups GA.. (8192: group 2): a device call after `whole` in the
                same invocation, on the same CTAs. `whole` stops at GA groups because
                the memory analysis of `schedule finalize` on 3 groups did not end
                in 1 h (2 groups: minutes)."""
                with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                    with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                        wgmma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                                  a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                        a = tf.reshape(x, (M, K))
                        for g in range(GA, GFULL):
                            for bn in range(NB):   # the column blocks in order: q|gate per head, k heads, v heads
                                for mi in range((cta.persistent - (g * NB + bn) * GROUP) % CTAS, GROUP, CTAS):
                                    m = (g * GROUP + mi) * BM
                                    n = bn * BN
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                                    for k in tf.tile(K, BK):
                                        with threads[2, :32] as _loader:
                                            lhs = tf.schedule((a[m:m + BM, k],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                            rhs = tf.schedule((w[n:n + BN, k],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                        with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                            acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                              op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        cs = tf.schedule((tf.cast(acc, "bf16"),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                    for _q in range((cta.persistent * 0 + bn) % NB, NQG, 1048576):   # once while bn is a q|gate block
                                        for _h in range((cta.persistent * 0 + bn) % 2, 1, 2):   # once for a q head (bn even); a gate head is stored as it lands
                                            with Mesh(threads[0:2, :], layout=(16, 16), names=("r", "c")) as e:
                                                x32 = tf.cast(tf.schedule((tf.reshape(cs, (16, 8, 16, 16)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 16))), "f32")
                                                sq = tf.schedule((x32 * x32,), op=T.copy(rmem_layout=((16 @ e.r, 8, 16 @ e.c, 16), (0, 16, 0, 1))))
                                                ms = tf.schedule((tf.reduce(sq, (2, 3), True, ReduceKind.MEAN),), op=T.copy(rmem_layout=(16 @ e.r, 8, 1, 1)))
                                                gw = tf.cast(tf.schedule((tf.reshape(gamma_q, (1, 1, 16, 16)),), op=T.copy(rmem_layout=(1, 1, 16 @ e.c, 16))), "f32") + 1.0
                                                nrm = tf.cast(x32 * tf.rsqrt(ms + EPS) * gw, "bf16")
                                                cs = tf.schedule((tf.reshape(nrm, (BM, BN)),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                                lo = tf.cast(tf.schedule((tf.reshape(cs[:, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                hi = tf.cast(tf.schedule((tf.reshape(cs[:, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                cl = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                ch = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                sl = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                sh = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                rlo = tf.cast(tf.cast(tf.cast(lo * cl, "bf16"), "f32") + tf.cast(tf.cast(-hi * sl, "bf16"), "f32"), "bf16")
                                                rhi = tf.cast(tf.cast(tf.cast(hi * ch, "bf16"), "f32") + tf.cast(tf.cast(lo * sh, "bf16"), "f32"), "bf16")
                                                cs = tf.insert_slice(cs, tf.reshape(rlo, (BM, HALF)), (0, 0))
                                                cs = tf.insert_slice(cs, tf.reshape(rhi, (BM, HALF)), (0, HALF))
                                    for _k0 in range(NQG - 1, (cta.persistent * 0 + bn) % NB, 1048576):   # once from the first k block ...
                                        for _k1 in range((cta.persistent * 0 + bn) % NB, NQG + NKH, 1048576):   # ... to the last: a k head (v heads are stored as they land)
                                            with Mesh(threads[0:2, :], layout=(16, 16), names=("r", "c")) as e:
                                                x32 = tf.cast(tf.schedule((tf.reshape(cs, (16, 8, 16, 16)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 16))), "f32")
                                                sq = tf.schedule((x32 * x32,), op=T.copy(rmem_layout=((16 @ e.r, 8, 16 @ e.c, 16), (0, 16, 0, 1))))
                                                ms = tf.schedule((tf.reduce(sq, (2, 3), True, ReduceKind.MEAN),), op=T.copy(rmem_layout=(16 @ e.r, 8, 1, 1)))
                                                gw = tf.cast(tf.schedule((tf.reshape(gamma_k, (1, 1, 16, 16)),), op=T.copy(rmem_layout=(1, 1, 16 @ e.c, 16))), "f32") + 1.0
                                                nrm = tf.cast(x32 * tf.rsqrt(ms + EPS) * gw, "bf16")
                                                cs = tf.schedule((tf.reshape(nrm, (BM, BN)),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                                lo = tf.cast(tf.schedule((tf.reshape(cs[:, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                hi = tf.cast(tf.schedule((tf.reshape(cs[:, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                cl = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                ch = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                sl = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                sh = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                                rlo = tf.cast(tf.cast(tf.cast(lo * cl, "bf16"), "f32") + tf.cast(tf.cast(-hi * sl, "bf16"), "f32"), "bf16")
                                                rhi = tf.cast(tf.cast(tf.cast(hi * ch, "bf16"), "f32") + tf.cast(tf.cast(lo * sh, "bf16"), "f32"), "bf16")
                                                cs = tf.insert_slice(cs, tf.reshape(rlo, (BM, HALF)), (0, 0))
                                                cs = tf.insert_slice(cs, tf.reshape(rhi, (BM, HALF)), (0, HALF))
                                    with threads[0, :32] as _storer:
                                        tile = tf.schedule((cs,), op=T.copy_async_tensor())
                                        out = tf.insert_slice(out, tile, (m, n))
                        return out

        if BLAST:
            @func
            def whole_last(out: Tensor[(M, N), "bf16"], x: Tensor[(1, M, K), "bf16"], w: Tensor[(N, K), "bf16"],
                           gamma_q: ConstTensor[(HD,), "bf16"], gamma_k: ConstTensor[(HD,), "bf16"],
                           cos: Tensor[(M, ROT), "bf16"], sin: Tensor[(M, ROT), "bf16"]):
                """The last M group's whole bands (where the plan has a tail): a
                device call after `whole` in the same invocation, on the same CTAs."""
                with Mesh(("cta",), layout=(CTAS,), names=("persistent",)) as cta:
                    with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                        wgmma = T.cuda.sm90.Wgmma(n=BN, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                                  a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                        a = tf.reshape(x, (M, K))
                        for bn in range(BLAST):   # the column blocks in order: q|gate per head, k heads, v heads
                            for mi in range((cta.persistent - (G0 * NB + bn) * GROUP) % CTAS, GROUP, CTAS):
                                m = (G0 * GROUP + mi) * BM
                                n = bn * BN
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ c.group, 8 @ c.lane8, 2, 4 @ c.warp, 2, 4 @ c.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                                for k in tf.tile(K, BK):
                                    with threads[2, :32] as _loader:
                                        lhs = tf.schedule((a[m:m + BM, k],), op=T.copy_async_tensor(smem_layout=Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                        rhs = tf.schedule((w[n:n + BN, k],), op=T.copy_async_tensor(smem_layout=Layout(((4, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3)), buffers=STAGES)
                                    with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                        acc = tf.schedule((acc, lhs, tf.bitcast(rhs, layout=Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))),
                                                          op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                                with Mesh(threads[0:2, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as c:
                                    cs = tf.schedule((tf.cast(acc, "bf16"),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                for _q in range((cta.persistent * 0 + bn) % NB, NQG, 1048576):   # once while bn is a q|gate block
                                    for _h in range((cta.persistent * 0 + bn) % 2, 1, 2):   # once for a q head (bn even); a gate head is stored as it lands
                                        with Mesh(threads[0:2, :], layout=(16, 16), names=("r", "c")) as e:
                                            x32 = tf.cast(tf.schedule((tf.reshape(cs, (16, 8, 16, 16)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 16))), "f32")
                                            sq = tf.schedule((x32 * x32,), op=T.copy(rmem_layout=((16 @ e.r, 8, 16 @ e.c, 16), (0, 16, 0, 1))))
                                            ms = tf.schedule((tf.reduce(sq, (2, 3), True, ReduceKind.MEAN),), op=T.copy(rmem_layout=(16 @ e.r, 8, 1, 1)))
                                            gw = tf.cast(tf.schedule((tf.reshape(gamma_q, (1, 1, 16, 16)),), op=T.copy(rmem_layout=(1, 1, 16 @ e.c, 16))), "f32") + 1.0
                                            nrm = tf.cast(x32 * tf.rsqrt(ms + EPS) * gw, "bf16")
                                            cs = tf.schedule((tf.reshape(nrm, (BM, BN)),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                            lo = tf.cast(tf.schedule((tf.reshape(cs[:, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            hi = tf.cast(tf.schedule((tf.reshape(cs[:, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            cl = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            ch = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            sl = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            sh = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            rlo = tf.cast(tf.cast(tf.cast(lo * cl, "bf16"), "f32") + tf.cast(tf.cast(-hi * sl, "bf16"), "f32"), "bf16")
                                            rhi = tf.cast(tf.cast(tf.cast(hi * ch, "bf16"), "f32") + tf.cast(tf.cast(lo * sh, "bf16"), "f32"), "bf16")
                                            cs = tf.insert_slice(cs, tf.reshape(rlo, (BM, HALF)), (0, 0))
                                            cs = tf.insert_slice(cs, tf.reshape(rhi, (BM, HALF)), (0, HALF))
                                for _k0 in range(NQG - 1, (cta.persistent * 0 + bn) % NB, 1048576):   # once from the first k block ...
                                    for _k1 in range((cta.persistent * 0 + bn) % NB, NQG + NKH, 1048576):   # ... to the last: a k head (v heads are stored as they land)
                                        with Mesh(threads[0:2, :], layout=(16, 16), names=("r", "c")) as e:
                                            x32 = tf.cast(tf.schedule((tf.reshape(cs, (16, 8, 16, 16)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 16))), "f32")
                                            sq = tf.schedule((x32 * x32,), op=T.copy(rmem_layout=((16 @ e.r, 8, 16 @ e.c, 16), (0, 16, 0, 1))))
                                            ms = tf.schedule((tf.reduce(sq, (2, 3), True, ReduceKind.MEAN),), op=T.copy(rmem_layout=(16 @ e.r, 8, 1, 1)))
                                            gw = tf.cast(tf.schedule((tf.reshape(gamma_k, (1, 1, 16, 16)),), op=T.copy(rmem_layout=(1, 1, 16 @ e.c, 16))), "f32") + 1.0
                                            nrm = tf.cast(x32 * tf.rsqrt(ms + EPS) * gw, "bf16")
                                            cs = tf.schedule((tf.reshape(nrm, (BM, BN)),), op=T.copy(smem_layout=Layout((128, (4, 64)), (64, (8192, 1))) | Swizzle(3, 4, 3)))
                                            lo = tf.cast(tf.schedule((tf.reshape(cs[:, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            hi = tf.cast(tf.schedule((tf.reshape(cs[:, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            cl = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            ch = tf.cast(tf.schedule((tf.reshape(cos[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            sl = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, 0:HALF], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            sh = tf.cast(tf.schedule((tf.reshape(sin[m:m + BM, HALF:ROT], (16, 8, 16, 2)),), op=T.copy(rmem_layout=(16 @ e.r, 8, 16 @ e.c, 2))), "f32")
                                            rlo = tf.cast(tf.cast(tf.cast(lo * cl, "bf16"), "f32") + tf.cast(tf.cast(-hi * sl, "bf16"), "f32"), "bf16")
                                            rhi = tf.cast(tf.cast(tf.cast(hi * ch, "bf16"), "f32") + tf.cast(tf.cast(lo * sh, "bf16"), "f32"), "bf16")
                                            cs = tf.insert_slice(cs, tf.reshape(rlo, (BM, HALF)), (0, 0))
                                            cs = tf.insert_slice(cs, tf.reshape(rhi, (BM, HALF)), (0, HALF))
                                with threads[0, :32] as _storer:
                                    tile = tf.schedule((cs,), op=T.copy_async_tensor())
                                    out = tf.insert_slice(out, tile, (m, n))
                        return out

        if HAS_TAIL:
            if GFULL > GA:
                @func
                def entry(hn: Tensor[(1, M, H), "bf16"], cos: Tensor[(M, ROT), "bf16"], sin: Tensor[(M, ROT), "bf16"],
                          w_q: ConstTensor[(HQ * 2 * HD, H), "bf16"], w_k: ConstTensor[(HKV * HD, H), "bf16"],
                          w_v: ConstTensor[(HKV * HD, H), "bf16"], gamma_q: ConstTensor[(HD,), "bf16"], gamma_k: ConstTensor[(HD,), "bf16"]):
                    w = pack(w_q, w_k, w_v)  # noqa: F821
                    ws = pieces(hn, w)  # noqa: F821
                    return whole_last(whole_b(whole(hn, w, gamma_q, gamma_k, cos, sin), hn, w, gamma_q, gamma_k, cos, sin), hn, w, gamma_q, gamma_k, cos, sin), ws  # noqa: F821
            else:
                @func
                def entry(hn: Tensor[(1, M, H), "bf16"], cos: Tensor[(M, ROT), "bf16"], sin: Tensor[(M, ROT), "bf16"],
                          w_q: ConstTensor[(HQ * 2 * HD, H), "bf16"], w_k: ConstTensor[(HKV * HD, H), "bf16"],
                          w_v: ConstTensor[(HKV * HD, H), "bf16"], gamma_q: ConstTensor[(HD,), "bf16"], gamma_k: ConstTensor[(HD,), "bf16"]):
                    w = pack(w_q, w_k, w_v)  # noqa: F821
                    ws = pieces(hn, w)  # noqa: F821
                    return whole_last(whole(hn, w, gamma_q, gamma_k, cos, sin), hn, w, gamma_q, gamma_k, cos, sin), ws  # noqa: F821
        else:
            @func
            def entry(hn: Tensor[(1, M, H), "bf16"], cos: Tensor[(M, ROT), "bf16"], sin: Tensor[(M, ROT), "bf16"],
                      w_q: ConstTensor[(HQ * 2 * HD, H), "bf16"], w_k: ConstTensor[(HKV * HD, H), "bf16"],
                      w_v: ConstTensor[(HKV * HD, H), "bf16"], gamma_q: ConstTensor[(HD,), "bf16"], gamma_k: ConstTensor[(HD,), "bf16"]):
                o = tf.reshape(whole(hn, pack(w_q, w_k, w_v), gamma_q, gamma_k, cos, sin), (1, M, N))  # noqa: F821
                qg = tf.reshape(o[:, :, 0:HQ * 2 * HD], (1, M, HQ, 2 * HD))
                kk = tf.reshape(o[:, :, HQ * 2 * HD:HQ * 2 * HD + HKV * HD], (1, M, HKV, HD))
                vv = tf.reshape(o[:, :, HQ * 2 * HD + HKV * HD:N], (1, M, HKV, HD))
                return qg[:, :, :, 0:HD], qg[:, :, :, HD:2 * HD], kk, vv

    QKV.__doc__ = f"qkv at seq_len={M}: {p['MB'] * NB} tiles, tail {p['REM']} x {S}"
    if not HAS_TAIL:
        return QKV, None

    @module(entry="entry", target=_H200, topologies=(Topology("cta", NBT * GROUP * CY), Topology("thread", 384)))
    class QKVTail:
        @func
        def tail(out: Tensor[(M, N), "bf16"], ws: Tensor[(UNITS * BM, BN), "f32"], gamma_q: ConstTensor[(HD,), "bf16"],
                 gamma_k: ConstTensor[(HD,), "bf16"], cos: Tensor[(M, ROT), "bf16"], sin: Tensor[(M, ROT), "bf16"]):
            with Mesh(("cta",), layout=(NBT, GROUP, CY), names=("tb", "mi", "cy")) as cta:
                with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                    with Mesh(threads[0:2, :], layout=(RT, TC), names=("row", "col")) as th:
                        t = cta.tb * GROUP + cta.mi
                        acc = tf.schedule((tf.reshape(ws[t * S * BM + cta.cy * RT:t * S * BM + cta.cy * RT + RT, :], (RT, BN // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(RT @ th.row, BN // (4 * TC), TC @ th.col, 4)))
                        for j in range(1, S):
                            acc = acc + tf.schedule((tf.reshape(ws[(t * S + j) * BM + cta.cy * RT:(t * S + j) * BM + cta.cy * RT + RT, :], (RT, BN // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(RT @ th.row, BN // (4 * TC), TC @ th.col, 4)))
                        m = (G0 * GROUP + cta.mi) * BM + cta.cy * RT
                        bn = B0 + cta.tb
                        cs = tf.cast(acc, "bf16")
                        for _h in range((cta.tb + NBT - TBK0) % NBT, NKT, NBT):   # once for a k-head band
                            x32 = tf.cast(cs, "f32")
                            sq = tf.schedule((x32 * x32,), op=T.copy(rmem_layout=((RT @ th.row, BN // (4 * TC), TC @ th.col, 4), (0, 4, 0, 1))))
                            ms = tf.schedule((tf.reduce(sq, (1, 2, 3), True, ReduceKind.MEAN),), op=T.copy(rmem_layout=(RT @ th.row, 1, 1, 1)))
                            gw = tf.cast(tf.schedule((tf.reshape(gamma_k, (1, BN // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(1, BN // (4 * TC), TC @ th.col, 4))), "f32") + 1.0
                            nrm = tf.cast(x32 * tf.rsqrt(ms + EPS) * gw, "bf16")
                            ns = tf.schedule((tf.reshape(nrm, (RT, BN)),), op=T.copy(smem_layout=Layout((RT, BN), (BN, 1))))
                            lo = tf.cast(tf.schedule((tf.reshape(ns[:, 0:HALF], (RT, HALF // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(RT @ th.row, HALF // (4 * TC), TC @ th.col, 4))), "f32")
                            hi = tf.cast(tf.schedule((tf.reshape(ns[:, HALF:ROT], (RT, HALF // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(RT @ th.row, HALF // (4 * TC), TC @ th.col, 4))), "f32")
                            cl = tf.cast(tf.schedule((tf.reshape(cos[m:m + RT, 0:HALF], (RT, HALF // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(RT @ th.row, HALF // (4 * TC), TC @ th.col, 4))), "f32")
                            ch = tf.cast(tf.schedule((tf.reshape(cos[m:m + RT, HALF:ROT], (RT, HALF // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(RT @ th.row, HALF // (4 * TC), TC @ th.col, 4))), "f32")
                            sl = tf.cast(tf.schedule((tf.reshape(sin[m:m + RT, 0:HALF], (RT, HALF // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(RT @ th.row, HALF // (4 * TC), TC @ th.col, 4))), "f32")
                            sh = tf.cast(tf.schedule((tf.reshape(sin[m:m + RT, HALF:ROT], (RT, HALF // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(RT @ th.row, HALF // (4 * TC), TC @ th.col, 4))), "f32")
                            rlo = tf.cast(tf.cast(tf.cast(lo * cl, "bf16"), "f32") + tf.cast(tf.cast(-hi * sl, "bf16"), "f32"), "bf16")
                            rhi = tf.cast(tf.cast(tf.cast(hi * ch, "bf16"), "f32") + tf.cast(tf.cast(lo * sh, "bf16"), "f32"), "bf16")
                            ns = tf.insert_slice(ns, tf.reshape(rlo, (RT, HALF)), (0, 0))
                            ns = tf.insert_slice(ns, tf.reshape(rhi, (RT, HALF)), (0, HALF))
                            cs = tf.schedule((tf.reshape(ns, (RT, BN // (4 * TC), TC, 4)),), op=T.copy(rmem_layout=(RT @ th.row, BN // (4 * TC), TC @ th.col, 4)))
                        return tf.insert_slice(out, tf.reshape(cs, (RT, BN)), (m, bn * BN))

        @func
        def entry(out: Tensor[(M, N), "bf16"], ws: Tensor[(UNITS * BM, BN), "f32"], gamma_q: ConstTensor[(HD,), "bf16"],
                  gamma_k: ConstTensor[(HD,), "bf16"], cos: Tensor[(M, ROT), "bf16"], sin: Tensor[(M, ROT), "bf16"]):
            o = tf.reshape(tail(out, ws, gamma_q, gamma_k, cos, sin), (1, M, N))  # noqa: F821
            qg = tf.reshape(o[:, :, 0:HQ * 2 * HD], (1, M, HQ, 2 * HD))
            kk = tf.reshape(o[:, :, HQ * 2 * HD:HQ * 2 * HD + HKV * HD], (1, M, HKV, HD))
            vv = tf.reshape(o[:, :, HQ * 2 * HD + HKV * HD:N], (1, M, HKV, HD))
            return qg[:, :, :, 0:HD], qg[:, :, :, HD:2 * HD], kk, vv

    QKVTail.__doc__ = f"qkv at seq_len={M}, invocation 2 of 2: the split tail ({p['REM']} tiles x {S} pieces)"
    return QKV, QKVTail


# One Module per (function, length) -- two where the plan has a split tail
# (`<Name><n>` then `<Name>Tail<n>`, two invocations in that order):
# `finalize scheduled/gemm.py:<Name>.<func>`.
def _register(name, M, mods):
    main, tail = mods
    globals()[f"{name}{M}"] = main
    if tail is not None:
        globals()[f"{name}Tail{M}"] = tail


for _M in (1024, 4096, 8192):
    _register("GateUp", _M, gate_up(_M))
    _register("InProj", _M, in_proj(_M))
    _register("Down", _M, residual("down", _M, I))
    _register("OutProj", _M, residual("out_proj", _M, 6144))
    _register("OProj", _M, residual("o_proj", _M, 6144))
    _register("QKV", _M, qkv(_M))
