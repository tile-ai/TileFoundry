"""Qwen3.5-35B-A3B's MoE block: the router, 256 routed experts, one shared expert.

Where the decode step's time goes
---------------------------------
Per token per layer this block reads 8 x 3 x 512 x 2048 bf16 of routed expert
weight (50 MB), 6.3 MB of shared expert and 1 MB of router -- 57 MB out of the
layer's ~62 MB, 40 times per token. Three measurements shape every choice below,
all of them made on this machine and all of them made **cold**:

* **cold is 3.2 TB/s, not 4.4.** A CUDA graph that replays one call 100 times
  reads the same 50 MB out of a 50 MB L2 on calls 2..100 and flatters this file
  by 30-40%: the gate/up kernel measures 8.7 us that way and 13.5 us when each
  replay reads a different copy of the weights, which is what a decode step does
  (5.9 GB per token, nothing resident). Every number in this file is the second
  kind. The plateau is ~3.2 TB/s once a kernel has >=128 blocks, and no tile,
  thread count or pipeline depth got past it.
* **a graph-replayed kernel node costs 1.6 us** (measured with a 32-thread kernel
  that writes one float). Against ~18 us of memory time that is a tax on kernel
  count, so `experts` is *two* kernels, not four: one produces every SwiGLU
  hidden, routed and shared, and one consumes them. Four kernels measure 37.3 us
  against 21.2 -- more than the 3.2 us of launches, because the shared expert
  alone is 9 blocks of work that cannot fill the machine on its own but slots
  into the routed kernel's grid for free.
* **`Tensor.zero_()` is a 1.2 us graph node.** The down kernel accumulates with
  atomics and so needs a zeroed output; the *previous* kernel's blocks zero it
  instead, 2048 stores spread over 73 blocks that were going to run anyway.

The gather
----------
`indices` is a device tensor -- the step is captured in a CUDA graph, so nothing
may come back to the host. Every expert-weight read is therefore addressed by a
device value: block (slot s, tile) reads `w_gate[Idx[0, s]]`, and the index lands
in the `T.copy` extent, uniform across the block. It costs nothing measurable
against the same kernel with a constant expert.

The dead rows are worth 440x accuracy, for free
-----------------------------------------------
`basic.py` puts the vector in row 0 of a 16-row MMA tile and leaves rows 1..15
uninitialised, because an MMA computes each output row from its own input row.
The consequence nobody wants to leave on the table: **row 1 is a free GEMV
against the same weight tile.** Rounding an f32 activation to bf16 costs 2.2e-3
of relative error -- the error budget for this whole file is 2e-3, and
`routed_experts` rounds twice -- so row 1 gets `x - bf16(x)` and the epilogue
adds `acc[0] + acc[1]`. That is a 17-bit activation for the price of one extra
shared-memory store per element: measured 2.16e-3 -> 4.9e-6 with the kernel time
unchanged (8.7 us against 9.1, inside the run-to-run noise). The router takes it
one row further (rows 0,1,2, ~f32) because its output is an *index*: see
`_router_logits`.

Doing the same thing the obvious way -- a second `T.gemm` on a second staged
vector -- costs 52% (8.7 -> 13.9 us at BN=32), because a 16xBNx16 wgmma at BN=32
runs at 1/8 of peak and the MMA stops hiding under the copies.

Weight dtypes
-------------
bf16 (production) and f32 (so the authored reference can be compared tightly)
are both accepted, and **f32 is accepted by splitting it, not by an f32 MMA**:
the tile is staged as bf16 hi + bf16 lo and multiplied by two MMAs. An f32
`T.gemm` on SM90 measures 8.5e-4 of relative error (it is a tf32 MMA) and needs
2x the shared memory for the staged operands, which does not fit the fused
kernel; the hi/lo split is 100x more accurate and fits in the same bytes. It is
slower -- the conversion cannot be a bulk async copy -- and that is the right
trade for a dtype only the reference comparison uses.
"""
from __future__ import annotations

import functools

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

try:
    from .basic import BM, rms_norm
except ImportError:  # `python kernels/moe.py`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kernels.basic import BM, rms_norm

#: `config.top_k`. Not derivable from any tensor `routing` is handed -- the
#: weights it returns are (S, K), which is the answer, not the question.
TOP_K = 8

#: Routed gate/up: 64 out-channels per block, so 512/64 x 8 slots = 64 blocks of
#: 512 KB (plus 8 for the shared expert and 1 for its gate: 73). Every one of
#: these was measured cold, at 37.8 MB (BN, blocks, TB/s):
#:   (32, 145, 2.38)  (64, 73, 2.81)  (128, 37, 2.00)
#: -- the opposite of what the same sweep says with a warm L2, where BN=32/145
#: blocks wins at 3.8 TB/s. Cold, a block's DRAM concurrency matters more than
#: the block count: BN=64 asks for 64 rows x 512 B per k-step instead of 32.
BN_H = 64

#: Down projection: 2048/64 x 8 slots = 256 blocks of 64 KB (plus 16 shared).
#: Cold at 18.9 MB: (64, 288, 2.63) (128, 144, 2.44) (256, 72, 2.29). Here the
#: reduction is only 512 long, so a block cannot ask for much at once and the
#: block count is what buys the concurrency.
BN_D, BK_D = 64, 128

THREADS = 128

#: The fused kernels branch on the block index into a `transpose_B` path (whose
#: tile rows are 256 B, so tilelang picks TMA) and a plain path (64 B rows, so it
#: picks `cp_async`). Warp specialisation then splits the block into a producer
#: warpgroup that only exists for the TMA branch and a consumer warpgroup that
#: runs the other one alone -- and the producer walks off the end of the kernel
#: while the consumer is still inside a whole-block barrier. The symptom is a
#: race, not a compile error: the shared expert's hidden came out 30% wrong on a
#: varying subset of tiles and its scalar gate came out `nan`, non-deterministic
#: run to run. Turning warp specialisation off makes every thread do both the
#: copies and the MMA, which is also *faster* here (15.7 us against 20.9).
_NO_WS = {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def _wdt(t: torch.Tensor) -> str:
    if t.dtype is torch.bfloat16:
        return "bfloat16"
    if t.dtype is torch.float32:
        return "float32"
    raise TypeError(f"weights must be bfloat16 or float32, got {t.dtype}")


def _stages(wdt: str) -> int:
    """Pipeline depth for the down kernel: 2 measured faster than 3 or 4 at every
    tile (7.18 us against 7.83 and 7.50), and an f32 weight is staged as two bf16
    tiles so it could not afford more anyway."""
    return 2


def _hcfg(wdt: str, mode: str) -> tuple[int, int]:
    """(BK, stages) for the hidden kernel.

    227 KB is all an SM will give a block, and this kernel spends 64 KB of it on
    the staged token. The fused mode carries four weight tiles (gate and up, in
    both orientations) where the others carry two, and an f32 weight doubles that
    again, so the pipeline gets shallower as the kernel gets busier. Overshooting
    is not a warning: the launch fails with "Failed to set the allowed dynamic
    shared memory size to 266240" -- which is what BK=128 with 3 stages asked for
    at BN=32, against the 227 KB an H200 SM allows.
    """
    if wdt == "float32":
        return (64, 2) if mode == "both" else (128, 2)
    return 256, 2


# ---------------------------------------------------------------- staging ----
# These are macros rather than inlined text because the same six lines appear at
# six call sites; `T.macro` gets the same source rewrite as `T.prim_func`, which
# is what makes `hi[i, j] = ...` legal inside one (a plain Python function raises
# "'Buffer' object does not support item assignment": the assignment rewriting
# only happens inside a transformed body).


@T.macro
def _stage2(hi, lo, src, r0, c0, D0, D1, split: bool):
    """`src[r0:r0+D0, c0:c0+D1]` into shared, as bf16."""
    if split:
        for i, j in T.Parallel(D0, D1):
            v = src[r0 + i, c0 + j]
            h = T.cast(v, "bfloat16")
            hi[i, j] = h
            lo[i, j] = T.cast(v - T.cast(h, "float32"), "bfloat16")
    else:
        T.copy(src[r0:r0 + D0, c0:c0 + D1], hi)


@T.macro
def _stage3(hi, lo, src, e, r0, c0, D0, D1, split: bool):
    """`src[e, r0:r0+D0, c0:c0+D1]` into shared, as bf16. `e` is the gather."""
    if split:
        for i, j in T.Parallel(D0, D1):
            v = src[e, r0 + i, c0 + j]
            h = T.cast(v, "bfloat16")
            hi[i, j] = h
            lo[i, j] = T.cast(v - T.cast(h, "float32"), "bfloat16")
    else:
        T.copy(src[e, r0:r0 + D0, c0:c0 + D1], hi)


@T.macro
def _split2(xs, v, j):
    """`v` into rows 0 and 1 of the staged vector: bf16 hi, then the residual.
    Row 1 is free MMA work -- see the module docstring."""
    r0 = T.cast(v, "bfloat16")
    xs[0, j] = r0
    xs[1, j] = T.cast(v - T.cast(r0, "float32"), "bfloat16")


# ---------------------------------------------------------------- router ----

#: How many blocks split the router's 1 MB GEMV. The top-K needs every logit in
#: one block and tilelang has no grid barrier that survives graph capture, so the
#: obvious shape -- the whole thing in one block -- is what this file shipped
#: first: **15.6 us**, because one SM sustains ~90 GB/s against a cold HBM and
#: 1 MB is 11 us of it. Sixteen blocks of 64 KB plus a second kernel that selects
#: is 7.4 us, and the second kernel's 1.6 us launch is most of what is left.
ROUTER_SPLIT = 16


@functools.lru_cache(maxsize=None)
def _router_logits(wdt: str, H: int, E: int, NS: int):
    """`Lp[c] = x[c-th chunk] @ W[c-th chunk]`, f32, one block per chunk.

    Three staged rows, not two: this GEMV's consumer is a *selection*, so an
    error that moves a logit can change which experts run. A bf16 activation
    moves one by ~2e-3 of its own size while the gap between the 8th and 9th
    largest of 256 is ~6e-2 of one -- one draw in thirty would pick a different
    expert. Rows 0,1,2 hold x to ~26 bits, which puts the logit error at the f32
    accumulation floor (~3e-6) and makes 220 draws agree with `torch.topk`
    exactly. The rows are free: the MMA computes all 16 of them anyway.

    Partials rather than atomics so that the sum is over `c` in order, the same
    order for every column. Two identical columns of `W` then give bitwise
    identical logits, which is what makes a tie a tie -- see `_router_select`.
    """
    CH = H // NS
    BK = min(CH, 64 if wdt == "bfloat16" else 32)
    ST = min(CH // BK, 4)
    THR = max(32, min(256, 32 * (E // 8)))

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((1, H), "float32"),
            W: T.Tensor((H, E), wdt),
            Lp: T.Tensor((NS, E), "float32"),
        ):
            # `wdt` is spelled out in the body below rather than hoisted into a
            # `split` flag on purpose: tilelang evaluates the (stringified)
            # annotations above against *this function's closure cells*, and
            # Python only creates a cell for a name the code references. A
            # dimension or dtype that appears only in an annotation raises
            # `NameError` (basic.py records the same trap).
            with T.Kernel(NS, threads=THR) as c:
                xs = T.alloc_shared((BM, CH), "bfloat16")
                wh = T.alloc_shared((BK, E), "bfloat16")
                wl = T.alloc_shared((BK, E), "bfloat16") if wdt == "float32" else wh
                acc = T.alloc_fragment((BM, E), "float32")
                out = T.alloc_shared((BM, E), "float32")
                for j in T.Parallel(CH):
                    # The `< H` guard is a bounds check and also the reason `H`
                    # resolves at all -- see the note above.
                    v = X[0, c * CH + j] if c * CH + j < H else 0.0
                    r0 = T.cast(v, "bfloat16")
                    d1 = v - T.cast(r0, "float32")
                    r1 = T.cast(d1, "bfloat16")
                    xs[0, j] = r0
                    xs[1, j] = r1
                    xs[2, j] = T.cast(d1 - T.cast(r1, "float32"), "bfloat16")
                T.clear(acc)
                T.sync_threads()
                for ko in T.Pipelined(CH // BK, num_stages=ST):
                    _stage2(wh, wl, W, c * CH + ko * BK, 0, BK, E,
                            wdt == "float32")
                    T.gemm(xs[:, ko * BK:(ko + 1) * BK], wh, acc)
                    if wdt == "float32":
                        T.gemm(xs[:, ko * BK:(ko + 1) * BK], wl, acc)
                T.copy(acc, out)
                for j in T.Parallel(E):
                    Lp[c, j] = out[0, j] + out[1, j] + out[2, j]

        return main

    return build()


@functools.lru_cache(maxsize=None)
def _router_select(E: int, K: int, NS: int):
    """softmax -> top-K -> renormalise, one block, from the logit partials.

    The selection is K rounds of one `reduce_max` over an int64 key,
    `bits(exp(l - max)) << 32 | (E-1-j)`: `exp(l - max)` is non-negative so its
    bit pattern sorts as an integer, and packing the complement of the index
    underneath means a single max returns the largest value *and*, among equals,
    the lowest index -- `torch.topk`'s selection rule in one reduction instead of
    a max followed by an argmin (measured 2.3 us against 3.7 for the alternative,
    counting ranks pairwise).

    The 256-wide softmax denominator is never formed: it divides every
    probability by the same number, so it changes neither the order nor which
    entries are equal, and the returned weights renormalise over the top K
    anyway. Only the max is needed, to keep `exp` in range.
    """
    #: One warp. This block does 1 + K whole-block reductions over E and nothing
    #: else, and `T.reduce_max` over a fragment one warp owns is a shuffle tree
    #: with no shared memory and no `__syncthreads`: the kernel measures 5.15 us
    #: with 256 threads and 4.31 with 32, both including a 1.6 us launch node.
    #: Each lane holds E/32 of the keys.
    THR = 32

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Lp: T.Tensor((NS, E), "float32"),
            Wt: T.Tensor((1, K), "float32"),
            Ind: T.Tensor((1, K), "int64"),
        ):
            with T.Kernel(1, threads=THR) as _:
                part = T.alloc_fragment((NS, E), "float32")
                lg = T.alloc_fragment((E,), "float32")
                red = T.alloc_fragment((E,), "float32")
                mx = T.alloc_fragment((1,), "float32")
                top = T.alloc_fragment((1,), "int64")
                lgmax = T.alloc_shared((1,), "float32")
                key = T.alloc_shared((E,), "int64")
                kf = T.alloc_fragment((E,), "int64")
                selv = T.alloc_shared((K,), "float32")
                seli = T.alloc_shared((K,), "int64")
                tot = T.alloc_shared((1,), "float32")

                # Every partial at once, then one reduction: a serial loop over
                # `c` around a `T.Parallel(E)` chains NS dependent global loads
                # and cost 3.2 us of pure latency on its own.
                for c, j in T.Parallel(NS, E):
                    part[c, j] = Lp[c, j]
                T.reduce_sum(part, lg, dim=0)
                for j in T.Parallel(E):
                    red[j] = lg[j]
                T.reduce_max(red, mx, dim=0, clear=True)
                if T.get_thread_binding() == 0:
                    lgmax[0] = mx[0]
                T.sync_threads()
                for j in T.Parallel(E):
                    key[j] = (T.cast(T.reinterpret(
                        "int32", T.exp(lg[j] - lgmax[0])), "int64") << 32) \
                        | T.cast(E - 1 - j, "int64")
                T.sync_threads()
                for r in T.serial(K):
                    for j in T.Parallel(E):
                        kf[j] = key[j]
                    T.reduce_max(kf, top, dim=0, clear=True)
                    if T.get_thread_binding() == 0:
                        i = E - 1 - T.cast(top[0] & T.cast(0xFFFFFFFF, "int64"),
                                           "int32")
                        selv[r] = T.reinterpret("float32",
                                                T.cast(top[0] >> 32, "int32"))
                        seli[r] = T.cast(i, "int64")
                        key[i] = T.cast(-1, "int64")  # below every real key
                    T.sync_threads()
                if T.get_thread_binding() == 0:
                    tot[0] = 0.0
                    for r in T.serial(K):
                        tot[0] += selv[r]
                    for r in T.serial(K):
                        Wt[0, r] = selv[r] / tot[0]
                        Ind[0, r] = seli[r]

        return main

    return build()


# ------------------------------------------------------------ the experts ----


def _h_grid(mode: str, K: int, I: int, IS: int) -> tuple[int, int, int]:
    """(routed blocks, shared blocks, total) for the hidden kernel."""
    nr = K * (I // BN_H) if mode != "shared" else 0
    ns = (IS // BN_H + 1) if mode != "routed" else 0  # +1: the scalar gate
    return nr, ns, nr + ns


@functools.lru_cache(maxsize=None)
def _h_kernel(mode: str, wdt: str, H: int, E: int, K: int, I: int, IS: int):
    """`h[s] = silu(w_gate[e_s] @ x) * (w_up[e_s] @ x)` for the K routed slots,
    and the same for the shared expert, and the shared expert's scalar gate.

    One kernel for all three because they read the same token and their outputs
    are all consumed by `_down_kernel`: three launches would cost 3.2 us more
    than one, which is a quarter of the block's memory time. The block index
    selects the job. The two orientations are the reason there is a branch at all
    rather than one uniform body: an expert's `w_gate[e]` is (out, in), so the MMA
    takes `transpose_B` and the copy walks BN rows of BK*2 bytes; the shared
    expert's `w_shared_gate` is (in, out), so it does not, and its tile is the
    transpose of the routed one -- a different shape, hence a second set of
    staging buffers.

    Blocks in the routed range also zero `Out`, which `_down_kernel` atomically
    accumulates into. Zeroing it here is free -- 2048 stores spread over 73
    blocks that are about to spend 13 us reading weights -- where a
    `Tensor.zero_()` graph node costs 1.2 us, a fifth of the down kernel.
    """
    BK, ST = _hcfg(wdt, mode)
    NR, NS, NB = _h_grid(mode, K, I, IS)
    NTR = I // BN_H
    KOR = H // BK
    CH = (H + NB - 1) // NB  # zeroing: this block's slice of Out

    # One signature serves all three modes: the tensors a mode never reads are
    # declared 1-element and the caller passes a dummy. They live in a dict
    # because a name that appears *only* in an annotation gets no closure cell
    # and raises `NameError` -- `SH` is read by `T.Kernel(SH["nb"], ...)` below,
    # which is what makes every shape in it resolve.
    SH = dict(
        nb=NB,
        ex=(E, I, H) if mode != "shared" else (1, 1, 1),
        k=K if mode != "shared" else 1,
        hr=(K, I) if mode != "shared" else (1, 1),
        sg=(H, IS) if mode != "routed" else (1, 1),
        ss=(H, 1) if mode != "routed" else (1, 1),
        hs=IS if mode != "routed" else 1,
    )

    @tilelang.jit(pass_configs=_NO_WS if mode == "both" else None)
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((1, H), "float32"),
            Idx: T.Tensor((1, SH["k"]), "int64"),
            WG: T.Tensor(SH["ex"], wdt),
            WU: T.Tensor(SH["ex"], wdt),
            WSG: T.Tensor(SH["sg"], wdt),
            WSU: T.Tensor(SH["sg"], wdt),
            WSS: T.Tensor(SH["ss"], wdt),
            HR: T.Tensor(SH["hr"], "float32"),
            HS: T.Tensor((SH["hs"],), "float32"),
            SC: T.Tensor((1,), "float32"),
            Out: T.Tensor((H,), "float32"),
        ):
            with T.Kernel(SH["nb"], threads=THREADS) as b:
                xs = T.alloc_shared((BM, H), "bfloat16")
                gh = T.alloc_shared((BN_H, BK), "bfloat16")
                gl = T.alloc_shared((BN_H, BK), "bfloat16") \
                    if wdt == "float32" else gh
                uh = T.alloc_shared((BN_H, BK), "bfloat16")
                ul = T.alloc_shared((BN_H, BK), "bfloat16") \
                    if wdt == "float32" else uh
                # (BK, BN) against the routed (BN, BK): the same bytes, not the
                # same buffer.
                sgh = T.alloc_shared((BK, BN_H), "bfloat16")
                sgl = T.alloc_shared((BK, BN_H), "bfloat16") \
                    if wdt == "float32" else sgh
                suh = T.alloc_shared((BK, BN_H), "bfloat16")
                sul = T.alloc_shared((BK, BN_H), "bfloat16") \
                    if wdt == "float32" else suh
                ag = T.alloc_fragment((BM, BN_H), "float32")
                au = T.alloc_fragment((BM, BN_H), "float32")
                og = T.alloc_shared((BM, BN_H), "float32")
                ou = T.alloc_shared((BM, BN_H), "float32")

                for j in T.Parallel(H):
                    _split2(xs, X[0, j], j)
                T.clear(ag)
                T.clear(au)
                if mode != "shared":
                    for j in T.Parallel(CH):
                        if b * CH + j < H:
                            Out[b * CH + j] = 0.0
                T.sync_threads()

                if mode != "shared":
                    if b < NR:
                        s = b // NTR
                        ti = b % NTR
                        e = Idx[0, s]
                        for ko in T.Pipelined(KOR, num_stages=ST):
                            _stage3(gh, gl, WG, e, ti * BN_H, ko * BK,
                                    BN_H, BK, wdt == "float32")
                            _stage3(uh, ul, WU, e, ti * BN_H, ko * BK,
                                    BN_H, BK, wdt == "float32")
                            T.gemm(xs[:, ko * BK:(ko + 1) * BK], gh, ag,
                                   transpose_B=True)
                            T.gemm(xs[:, ko * BK:(ko + 1) * BK], uh, au,
                                   transpose_B=True)
                            if wdt == "float32":
                                T.gemm(xs[:, ko * BK:(ko + 1) * BK], gl, ag,
                                       transpose_B=True)
                                T.gemm(xs[:, ko * BK:(ko + 1) * BK], ul, au,
                                       transpose_B=True)
                        T.copy(ag, og)
                        T.copy(au, ou)
                        for j in T.Parallel(BN_H):
                            g = og[0, j] + og[1, j]
                            u = ou[0, j] + ou[1, j]
                            HR[s, ti * BN_H + j] = g / (1.0 + T.exp(-g)) * u

                if mode != "routed":
                    if NR <= b:
                        if b < NB - 1:
                            ti = b - NR
                            for ko in T.Pipelined(KOR, num_stages=ST):
                                _stage2(sgh, sgl, WSG, ko * BK, ti * BN_H,
                                        BK, BN_H, wdt == "float32")
                                _stage2(suh, sul, WSU, ko * BK, ti * BN_H,
                                        BK, BN_H, wdt == "float32")
                                T.gemm(xs[:, ko * BK:(ko + 1) * BK], sgh, ag)
                                T.gemm(xs[:, ko * BK:(ko + 1) * BK], suh, au)
                                if wdt == "float32":
                                    T.gemm(xs[:, ko * BK:(ko + 1) * BK], sgl, ag)
                                    T.gemm(xs[:, ko * BK:(ko + 1) * BK], sul, au)
                            T.copy(ag, og)
                            T.copy(au, ou)
                            for j in T.Parallel(BN_H):
                                g = og[0, j] + og[1, j]
                                u = ou[0, j] + ou[1, j]
                                HS[ti * BN_H + j] = g / (1.0 + T.exp(-g)) * u
                        else:
                            # The shared expert's own gate: one scalar, so one
                            # block and a plain reduction rather than an MMA.
                            # 4 KB of weight and ~1 us of latency, hidden under
                            # the 72 blocks that are reading 37 MB.
                            dot = T.alloc_fragment((H,), "float32")
                            acc1 = T.alloc_fragment((1,), "float32")
                            for j in T.Parallel(H):
                                dot[j] = X[0, j] * T.cast(WSS[j, 0], "float32")
                            T.reduce_sum(dot, acc1, dim=0)
                            if T.get_thread_binding() == 0:
                                SC[0] = 1.0 / (1.0 + T.exp(-acc1[0]))

        return main

    return build()


def _down_grid(mode: str, H: int, K: int) -> tuple[int, int, int]:
    nt = H // BN_D
    nr = nt * K if mode != "shared" else 0
    ns = nt if mode != "routed" else 0
    return nr, ns, nr + ns


@functools.lru_cache(maxsize=None)
def _down_kernel(mode: str, wdt: str, H: int, E: int, K: int, I: int, IS: int):
    """`out = sum_s weights[s] * (w_down[e_s] @ h[s]) + gate * (h_s @ w_sd)`.

    One block per (output tile, slot) and `atomic_add` into a zeroed output, not
    one block per output tile looping the 8 slots. Cold, on the routed half's
    16.8 MB, slots per block against blocks and time:
        8 slots  32 blocks  20.82 us   (the loop; 16 blocks at BN=128: 26.9)
        4 slots  64 blocks  11.61 us
        2 slots 128 blocks   8.11 us
        1 slot  256 blocks   6.65 us   <- this
    **3.1x**, and it is all block count: the reduction here is only 512 long, so
    a block cannot keep much memory in flight and the only way to get concurrency
    is more blocks. Zeroing `Out` in `_h_kernel` rather than with a
    `Tensor.zero_()` graph node is worth another 1.2 us (6.65 against 7.87).

    The routing weight is folded into the staged vector rather than the
    epilogue, so one accumulator serves the whole (slot, k) space, and the
    shared expert's scalar gate is folded into its own blocks' contribution --
    which is exactly right, since the gate multiplies only the shared half.

    A serial loop over slots *around* `T.Pipelined` does not compile:
    "ProducerConsumerWS: failed to replace pipeline loop". The 8-slot variant
    quoted above needed `tl.disable_warp_specialized`.
    """
    ST = _stages(wdt)
    NR, NS, NB = _down_grid(mode, H, K)
    NT = H // BN_D
    KOD = I // BK_D

    # See `_h_kernel`: one signature, mode-dependent shapes, in a dict so that
    # the annotations resolve against a name the body reads.
    SH = dict(
        nb=NB,
        ex=(E, H, I) if mode != "shared" else (1, 1, 1),
        k=K if mode != "shared" else 1,
        hr=(K, I) if mode != "shared" else (1, 1),
        sd=(IS, H) if mode != "routed" else (1, 1),
        hs=IS if mode != "routed" else 1,
        xw=I if mode != "shared" else IS,
        h=H,
    )

    @tilelang.jit(pass_configs=_NO_WS if mode == "both" else None)
    def build():
        @T.prim_func
        def main(
            HR: T.Tensor(SH["hr"], "float32"),
            Wt: T.Tensor((1, SH["k"]), "float32"),
            Idx: T.Tensor((1, SH["k"]), "int64"),
            WD: T.Tensor(SH["ex"], wdt),
            WSD: T.Tensor(SH["sd"], wdt),
            HS: T.Tensor((SH["hs"],), "float32"),
            SC: T.Tensor((1,), "float32"),
            Out: T.Tensor((SH["h"],), "float32"),
        ):
            with T.Kernel(SH["nb"], threads=THREADS) as b:
                xs = T.alloc_shared((BM, SH["xw"]), "bfloat16")
                dh = T.alloc_shared((BN_D, BK_D), "bfloat16")
                dl = T.alloc_shared((BN_D, BK_D), "bfloat16") \
                    if wdt == "float32" else dh
                sh = T.alloc_shared((BK_D, BN_D), "bfloat16")
                sl = T.alloc_shared((BK_D, BN_D), "bfloat16") \
                    if wdt == "float32" else sh
                acc = T.alloc_fragment((BM, BN_D), "float32")
                out = T.alloc_shared((BM, BN_D), "float32")
                T.clear(acc)

                if mode != "shared":
                    if b < NR:
                        # Tile-major so that the 8 blocks sharing an output tile
                        # are not issued back to back; their atomics collide.
                        s = b // NT
                        ti = b % NT
                        e = Idx[0, s]
                        w = Wt[0, s]
                        for j in T.Parallel(I):
                            _split2(xs, HR[s, j] * w, j)
                        T.sync_threads()
                        for ko in T.Pipelined(KOD, num_stages=ST):
                            _stage3(dh, dl, WD, e, ti * BN_D, ko * BK_D,
                                    BN_D, BK_D, wdt == "float32")
                            T.gemm(xs[:, ko * BK_D:(ko + 1) * BK_D], dh, acc,
                                   transpose_B=True)
                            if wdt == "float32":
                                T.gemm(xs[:, ko * BK_D:(ko + 1) * BK_D], dl, acc,
                                       transpose_B=True)
                        T.copy(acc, out)
                        for j in T.Parallel(BN_D):
                            T.atomic_add(Out[ti * BN_D + j], out[0, j] + out[1, j])

                if mode != "routed":
                    if NR <= b:
                        ti = b - NR
                        for j in T.Parallel(IS):
                            _split2(xs, HS[j], j)
                        T.sync_threads()
                        for ko in T.Pipelined(IS // BK_D, num_stages=ST):
                            _stage2(sh, sl, WSD, ko * BK_D, ti * BN_D,
                                    BK_D, BN_D, wdt == "float32")
                            T.gemm(xs[:, ko * BK_D:(ko + 1) * BK_D], sh, acc)
                            if wdt == "float32":
                                T.gemm(xs[:, ko * BK_D:(ko + 1) * BK_D], sl, acc)
                        T.copy(acc, out)
                        for j in T.Parallel(BN_D):
                            v = (out[0, j] + out[1, j]) * SC[0]
                            if mode == "shared":
                                Out[ti * BN_D + j] = v  # sole writer
                            else:
                                T.atomic_add(Out[ti * BN_D + j], v)

        return main

    return build()


# ------------------------------------------------------------ entry points ----

_DUMMY: dict = {}


def _dummy(shape, dtype, device):
    """A 1-element stand-in for a tensor this mode's kernel never reads. Cached
    at module scope so its address is stable across a graph capture."""
    key = (shape, dtype, device)
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(shape, dtype=dtype, device=device)
    return _DUMMY[key]


def post_norm(hidden: torch.Tensor, gamma_post: torch.Tensor) -> torch.Tensor:
    """`rms_norm(hidden, gamma_post)` as (1, H). eps 1e-6 is baked in."""
    H = hidden.shape[-1]
    out = torch.empty((1, H), device=hidden.device, dtype=torch.float32)
    rms_norm(H)(hidden.view(H), gamma_post, out.view(H))
    return out


def routing(tokens: torch.Tensor, w_router: torch.Tensor):
    """(weights (1, TOP_K) f32, indices (1, TOP_K) int64)."""
    H, E = w_router.shape
    dev = tokens.device
    ns = ROUTER_SPLIT if H % ROUTER_SPLIT == 0 else 1
    weights = torch.empty((1, TOP_K), device=dev, dtype=torch.float32)
    indices = torch.empty((1, TOP_K), device=dev, dtype=torch.int64)
    logits = torch.empty((ns, E), device=dev, dtype=torch.float32)
    _router_logits(_wdt(w_router), H, E, ns)(tokens, w_router, logits)
    _router_select(E, TOP_K, ns)(logits, weights, indices)
    return weights, indices


def routed_experts(tokens, weights, indices, w_gate, w_up, w_down) -> torch.Tensor:
    """The 8 selected experts, mixed by `weights`. (1, H) f32."""
    E, I, H = w_gate.shape
    K = weights.shape[1]
    wdt = _wdt(w_gate)
    dev = tokens.device
    out = torch.empty((1, H), device=dev, dtype=torch.float32)
    hr = torch.empty((K, I), device=dev, dtype=torch.float32)
    sc = _dummy((1,), torch.float32, dev)
    d2 = _dummy((1, 1), torch.__dict__[wdt], dev)
    d1 = _dummy((1,), torch.float32, dev)
    _h_kernel("routed", wdt, H, E, K, I, 1)(
        tokens, indices, w_gate, w_up, d2, d2, d2, hr, d1, sc, out.view(H))
    _down_kernel("routed", wdt, H, E, K, I, 1)(
        hr, weights, indices, w_down, d2, d1, sc, out.view(H))
    return out


def shared_expert(tokens, w_shared_gate, w_shared_up, w_shared_down,
                  w_shared_scale) -> torch.Tensor:
    """The dense expert every token goes through, times its own scalar gate."""
    H, IS = w_shared_gate.shape
    wdt = _wdt(w_shared_gate)
    dev = tokens.device
    out = torch.empty((1, H), device=dev, dtype=torch.float32)
    hs = torch.empty((IS,), device=dev, dtype=torch.float32)
    sc = torch.empty((1,), device=dev, dtype=torch.float32)
    d3 = _dummy((1, 1, 1), torch.__dict__[wdt], dev)
    d2 = _dummy((1, 1), torch.__dict__[wdt], dev)
    di = _dummy((1, 1), torch.int64, dev)
    df = _dummy((1, 1), torch.float32, dev)
    _h_kernel("shared", wdt, H, 1, 1, 1, IS)(
        tokens, di, d3, d3, w_shared_gate, w_shared_up, w_shared_scale,
        df, hs, sc, out.view(H))
    _down_kernel("shared", wdt, H, 1, 1, 1, IS)(
        df, df, di, d3, w_shared_down, hs, sc, out.view(H))
    return out


def experts(tokens, weights, indices, w_gate, w_up, w_down,
            w_shared_gate, w_shared_up, w_shared_down, w_shared_scale):
    """routed + shared, (1, 1, H) f32, in two kernels.

    The shared expert's gate/up joins the routed gate/up kernel (same token, 16
    more blocks) and its down joins the routed down kernel (16 more blocks,
    accumulating into the same output), so the whole block is two launches
    instead of four -- 3.2 us of graph nodes saved, plus the shared expert's
    small grids stop being their own latency.
    """
    E, I, H = w_gate.shape
    K = weights.shape[1]
    IS = w_shared_gate.shape[1]
    wdt = _wdt(w_gate)
    dev = tokens.device
    out = torch.empty((1, 1, H), device=dev, dtype=torch.float32)
    hr = torch.empty((K, I), device=dev, dtype=torch.float32)
    hs = torch.empty((IS,), device=dev, dtype=torch.float32)
    sc = torch.empty((1,), device=dev, dtype=torch.float32)
    _h_kernel("both", wdt, H, E, K, I, IS)(
        tokens, indices, w_gate, w_up, w_shared_gate, w_shared_up,
        w_shared_scale, hr, hs, sc, out.view(H))
    _down_kernel("both", wdt, H, E, K, I, IS)(
        hr, weights, indices, w_down, w_shared_down, hs, sc, out.view(H))
    return out


__all__ = ["experts", "post_norm", "routed_experts", "routing", "shared_expert"]


# ------------------------------------------------------------------ tests ----

if __name__ == "__main__":
    import time

    # Everything below goes through the *package* import, not this script's own
    # namespace. tilelang resolves each kernel's stringified annotations against
    # `func.__globals__` plus the closure cells, so a name this file happens to
    # define at module scope when run as `python kernels/moe.py` would paper over
    # a missing cell that `import kernels.moe` exposes. Ask for the module the
    # integration asks for.
    from kernels import moe

    def graph_bench(call, reps=100, iters=20):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                call()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                call()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters / reps * 1e6

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().max()).item()

    def main():
        torch.manual_seed(0)
        dev = "cuda"
        H, E, K, I, IS = 2048, 256, moe.TOP_K, 512, 512
        F = torch.nn.functional

        x = torch.randn(1, 1, H, device=dev)
        gamma = torch.randn(H, device=dev)
        tok = moe.post_norm(x, gamma)
        ref = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
               * gamma).view(1, H)
        print("=" * 74)
        print("(a) every entry point vs plain torch, BOTH weight dtypes")
        print(f"  post_norm (f32 only)          {rel(tok, ref):.3e}   [gate 1e-5]")

        for dt in (torch.bfloat16, torch.float32):
            def mk(*shape, s=0.05):
                return (torch.randn(*shape, device=dev) * s).to(dt)

            wr = mk(H, E)
            wg, wu, wd = mk(E, I, H), mk(E, I, H), mk(E, H, I)
            wsg, wsu, wsd, wss = mk(H, IS), mk(H, IS), mk(IS, H), mk(H, 1)
            wgt, idx = moe.routing(tok, wr)
            probs = torch.softmax(tok @ wr.float(), dim=-1)
            rv, ri = torch.topk(probs, K, dim=-1)
            rv = rv / rv.sum(-1, keepdim=True)

            def torch_routed(w, i):
                acc = torch.zeros(1, H, device=dev)
                for s in range(K):
                    e = int(i[0, s])
                    g = wg[e].float() @ tok[0]
                    u = wu[e].float() @ tok[0]
                    acc += (wd[e].float() @ (F.silu(g) * u)) * w[0, s]
                return acc

            def torch_shared():
                g = tok @ wsg.float()
                u = tok @ wsu.float()
                return ((F.silu(g) * u) @ wsd.float()
                        * torch.sigmoid(tok @ wss.float()))

            tr, ts = torch_routed(wgt, idx), torch_shared()
            print(f"  --- weights {str(dt).split('.')[1]:9s}          "
                  f"[gate 2e-3]")
            print(f"  routing weights               {rel(wgt, rv):.3e}")
            print(f"  routing indices               "
                  f"{'EXACT' if bool((idx == ri).all()) else 'MISMATCH'}")
            print(f"  routed_experts                "
                  f"{rel(moe.routed_experts(tok, wgt, idx, wg, wu, wd), tr):.3e}")
            print(f"  shared_expert                 "
                  f"{rel(moe.shared_expert(tok, wsg, wsu, wsd, wss), ts):.3e}")
            got = moe.experts(tok, wgt, idx, wg, wu, wd, wsg, wsu, wsd, wss)
            print(f"  experts                       "
                  f"{rel(got, (tr + ts).view(1, 1, H)):.3e}")
            del wg, wu, wd, wsg, wsu, wsd, wss, wr
            torch.cuda.empty_cache()

        # ---- top-k selection against torch.topk -------------------------
        print("=" * 74)
        print("top-8-of-256 selection vs torch.topk")
        bad = 0
        for trial in range(220):
            t = torch.randn(1, H, device=dev)
            w = (torch.randn(H, E, device=dev) * 0.05).bfloat16()
            _, i = moe.routing(t, w)
            _, ri = torch.topk(torch.softmax(t @ w.float(), dim=-1), K, dim=-1)
            if not bool((i == ri).all()):
                bad += 1
                if bad <= 3:
                    lg = (t @ w.float())[0].sort(descending=True).values
                    print(f"  draw {trial}: mine {i[0].tolist()} torch "
                          f"{ri[0].tolist()} gap(8,9)="
                          f"{float(lg[K - 1] - lg[K]):.2e}")
        print(f"  220 random draws:             {220 - bad}/220 exactly equal")
        # A tie is built by repeating three columns of the router: identical
        # columns give bitwise identical logits in any implementation, so both
        # sides see exactly three distinct values and the top 8 are all equal.
        cols = (torch.randn(3, H, device=dev) * 0.05).bfloat16()
        tw = cols[torch.arange(E, device=dev) % 3].T.contiguous()
        tt = torch.randn(1, H, device=dev)
        mv, mi = moe.routing(tt, tw)
        lg = tt @ tw.float()
        tv, ri = torch.topk(torch.softmax(lg, dim=-1), K, dim=-1)
        print(f"  tied draw ({int(lg.unique().numel())} distinct logits, "
              f"{int((lg == lg.max()).sum())} at the max):")
        print(f"    mine  {mi[0].tolist()}  weights spread "
              f"{float(mv.max() - mv.min()):.1e}")
        print(f"    torch {ri[0].tolist()}")
        print(f"    same set {sorted(mi[0].tolist()) == sorted(ri[0].tolist())}"
              f", mine index-ascending "
              f"{mi[0].tolist() == sorted(mi[0].tolist())}")
        #: `torch.topk` on CUDA *selects* the lowest indices among equals -- so
        #: the set of experts that run agrees -- but it does not return the tied
        #: group in index order: its sort is not stable. Exact tensor equality is
        #: therefore not a property any implementation can have on a tie; the set
        #: is, and within a tied group every weight is equal, so any order of the
        #: pairs gives the same block output.

        # ---- (b) against the authored Module ----------------------------
        print("=" * 74)
        import config as cfg
        import model
        from tilefoundry.runtime import DictResource

        # The published E, at the declared f32: 3.0 GiB of weights and 4.0 GiB
        # of peak through the interpreter, 0.9 s. `model.build(REAL.replace(
        # n_experts=32))` is the fallback if that ever stops fitting.
        EA = E
        mod = (model.Qwen3_5MoE if EA == E else
               model.build(cfg.REAL.replace(n_experts=EA))["Qwen3_5MoE"])
        w32 = dict(
            w_gate=torch.randn(EA, I, H, device=dev) * 0.05,
            w_up=torch.randn(EA, I, H, device=dev) * 0.05,
            w_down=torch.randn(EA, H, I, device=dev) * 0.05,
            w_shared_gate=torch.randn(H, IS, device=dev) * 0.05,
            w_shared_up=torch.randn(H, IS, device=dev) * 0.05,
            w_shared_down=torch.randn(IS, H, device=dev) * 0.05,
            w_shared_scale=torch.randn(H, 1, device=dev) * 0.05,
            gamma_post=gamma,
        )
        w32["router.w_router"] = torch.randn(H, EA, device=dev) * 0.05
        loaded = mod.load(DictResource(w32))
        t0 = time.time()
        aw, ai = loaded.router.routing(tok)
        print(f"(b) authored Module, f32 weights, E={EA}      "
              f"[routing {time.time() - t0:.1f}s]")
        mw, mi = moe.routing(tok, w32["router.w_router"])
        print(f"  routing weights               {rel(mw, aw):.3e}")
        print(f"  routing indices               "
              f"{'EXACT' if bool((mi == ai.view(1, K)).all()) else 'MISMATCH'}"
              f"  {mi[0].tolist()}")
        print(f"  post_norm                     "
              f"{rel(moe.post_norm(x, gamma), loaded.post_norm(x).view(1, H)):.3e}")
        t0 = time.time()
        aex = loaded.experts(tok, aw, ai)
        ai2 = ai.view(1, K).contiguous()
        gex = moe.experts(tok, aw, ai2, w32["w_gate"], w32["w_up"], w32["w_down"],
                          w32["w_shared_gate"], w32["w_shared_up"],
                          w32["w_shared_down"], w32["w_shared_scale"])
        print(f"  experts                       {rel(gex, aex.view(1, 1, H)):.3e}"
              f"   [{time.time() - t0:.1f}s]")
        print(f"  routed_experts                "
              f"{rel(moe.routed_experts(tok, aw, ai2, w32['w_gate'], w32['w_up'], w32['w_down']), loaded.routed_experts(tok, aw, ai)):.3e}")
        print(f"  shared_expert                 "
              f"{rel(moe.shared_expert(tok, w32['w_shared_gate'], w32['w_shared_up'], w32['w_shared_down'], w32['w_shared_scale']), loaded.shared_expert(tok)):.3e}")
        del w32, loaded, mod
        torch.cuda.empty_cache()

        # ---- (c) wall time, full size, bf16, cold ------------------------
        print("=" * 74)
        #: Every weight is allocated NCOPY times and call i reads copy i % NCOPY,
        #: and the 8 experts are re-drawn per call as well. Without that, a graph
        #: of 100 identical calls reads 50 MB out of a 50 MB L2 on calls 2..100
        #: and every kernel here looks ~30% faster than it is in a decode step,
        #: where the working set is 5.9 GB per token and nothing is resident.
        NCOPY, nrep = 8, 100
        print(f"(c) wall time per call in a CUDA graph, 256 experts bf16, "
              f"{NCOPY} rotated weight copies (cold L2)")
        mkb = lambda *s: (torch.randn(*s, device=dev) * 0.05).bfloat16()
        WR = [mkb(H, E) for _ in range(NCOPY)]
        WG = [mkb(E, I, H) for _ in range(NCOPY)]
        WU = [mkb(E, I, H) for _ in range(NCOPY)]
        WD = [mkb(E, H, I) for _ in range(NCOPY)]
        SG = [mkb(H, IS) for _ in range(NCOPY)]
        SU = [mkb(H, IS) for _ in range(NCOPY)]
        SD = [mkb(IS, H) for _ in range(NCOPY)]
        SS = [mkb(H, 1) for _ in range(NCOPY)]
        IDX = [torch.randperm(E, device=dev)[:K].unsqueeze(0).contiguous()
               for _ in range(nrep)]
        wgt = torch.full((1, K), 1.0 / K, device=dev)
        n = [0]

        def rot(fn):
            def call():
                c = n[0] % NCOPY
                fn(c, IDX[n[0] % nrep])
                n[0] += 1
            return call

        for name, call in (
            ("post_norm", lambda: moe.post_norm(x, gamma)),
            ("routing", rot(lambda c, i: moe.routing(tok, WR[c]))),
            ("routed_experts",
             rot(lambda c, i: moe.routed_experts(tok, wgt, i, WG[c], WU[c], WD[c]))),
            ("shared_expert",
             rot(lambda c, i: moe.shared_expert(tok, SG[c], SU[c], SD[c], SS[c]))),
            ("experts (2 kernels)",
             rot(lambda c, i: moe.experts(tok, wgt, i, WG[c], WU[c], WD[c],
                                         SG[c], SU[c], SD[c], SS[c]))),
            ("routed+shared (4 kernels)",
             rot(lambda c, i: moe.routed_experts(tok, wgt, i, WG[c], WU[c], WD[c])
                 + moe.shared_expert(tok, SG[c], SU[c], SD[c], SS[c]))),
            ("post_norm+routing+experts",
             rot(lambda c, i: moe.experts(moe.post_norm(x, gamma),
                                          *moe.routing(tok, WR[c]),
                                          WG[c], WU[c], WD[c],
                                          SG[c], SU[c], SD[c], SS[c]))),
        ):
            graph_bench(call, reps=20, iters=3)     # clock ramp
            print(f"  {name:27s} {graph_bench(call, reps=nrep, iters=20):7.2f} us")

    main()
