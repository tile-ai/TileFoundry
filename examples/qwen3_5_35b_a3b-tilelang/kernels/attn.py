"""`Qwen3_5FullAttention`: partial RoPE, and the whole GQA mixer step, fused.

The decode step is 54.5 MB of weight traffic (w_qg 33.5 MB, w_k + w_v 4.2 MB,
w_o 16.8 MB) against ~6 GFLOP, so this file is a memory-traffic problem with an
attention kernel bolted on. It is spent in **three** launches:

    1  `_proj_kernel`   rms_norm(hidden) -> {q||gate, k, v}   37.7 MB
    2  `_attn_kernel`   q/k head-norm + RoPE, softmax over the cache
    3  `_out_kernel`    log-sum-exp merge + gate + o_proj      16.8 MB

Why three and not six. `rms_norm` is 8 KB of work; launched on its own it would
cost more in launch latency than in traffic, so every block of kernel 1 computes
it redundantly (16 KB of reads per block against 294 KB of weights). The two
head norms and the RoPE are 4 KB of work on 16 heads; kernel 2 already has one
block per (kv head, context split) and each block needs the whole of its own q
and k, so it recomputes both -- again cheaper than a launch. The gate multiply
and the log-sum-exp merge of the context splits are kernel 3's prologue for the
same reason. What is left is exactly the three places where a weight matrix has
to be streamed, plus the attention, which cannot be fused into either
neighbour because it needs every projection finished and every score finished.

Why `basic.gemv` is not called
-----------------------------
Its signature is `(X, W, Y)`. There is nowhere in it to put the gamma, the
second and third weight matrix that share the same normalised vector, the
per-head RoPE, or the split-K merge -- and putting them anywhere else costs a
launch each, which is the whole budget. So the GEMV *idiom* is imported (BM, the
15-dead-rows trick, the shared-memory epilogue) and re-instantiated with the
prologue and epilogue this boundary needs. Everything basic.py's docstring
records still applies here and was re-measured: a fragment-slice epilogue is
still rejected, so row 0 is still reached through shared memory.

Two extra tricks on top of basic.py's
-------------------------------------
**Rows 0 and 1, not row 0.** basic.py puts the vector in row 0 of the 16-row MMA
tile and leaves 1..15 uninitialised. Rounding an f32 activation to bf16 costs
2^-9 relative *per element*, which after a K=2048 dot product is ~1.2e-3 of the
result -- right at the 2e-3 tolerance, with no margin. So row 0 holds
`bf16(x)` and row 1 holds `bf16(x - bf16(x))`; one MMA then produces both
`x_hi @ W` and `x_lo @ W`, and the epilogue adds them. 16 bits of mantissa on
the activation for *zero* extra traffic and zero extra MMAs, out of the 15 rows
that were being thrown away anyway. Measured relative error drops from ~1e-3 to
~3e-6.

The same trick pays twice more in kernel 2, where q sits in rows 0..7 (eight
query heads of one GQA group) and its low halves in rows 8..15, and where the
softmax probabilities sit in rows 0..7 / 8..15 of the PV operand. K and V come
out of an f32 cache, so they are split into two bf16 shared tiles and fed as two
accumulating `T.gemm`s -- the arithmetic is free at 6 GFLOP against 52 MB.

**One block reads ~20 GB/s, so the grid is everything.** This was the single
biggest lever measured. A block's own bandwidth saturates around 20 GB/s
whatever its pipeline depth (8 blocks on the 4.2 MB k/v pair: 30 us; 128 blocks:
4.1 us), so the launch time is set by how many blocks are resident and how
evenly the bytes fall across them, and going past the 132 SMs costs a whole
extra pass. Two consequences run through this file:

  - every grid here is <= 132 blocks, and 128 wherever the shape allows;
  - w_k and w_v are *not* their own blocks. At BN=128 they are 16 column tiles;
    16 blocks at 20 GB/s each are 13 us of critical path and the projection
    launch went from 11.0 to 21.9 us. Instead each of w_qg's 128 blocks carries
    one extra 128x128 tile of w_k or w_v -- 16 row-tiles x 4 column-tiles x 2
    matrices is exactly 128 tiles -- issued *before* the w_qg pipeline so its
    copy overlaps, consumed after it. +12% bytes per block, no extra blocks.

Two `T.Pipelined` loops cannot share one shared-memory tile (tilelang
multi-buffers per loop and the second one fails the shape check, quoted in the
report), so the bolted-on tile has its own buffer and no pipeline of its own.

A note on measurement
---------------------
w_qg (33.5 MB) and w_o (16.8 MB) both fit in H200's 60 MB L2. Any benchmark that
replays one layer in a loop therefore measures L2 bandwidth, not HBM. The
self-test reports both a `warm` number (one weight set, the prescribed harness)
and a `cold` one (eight rotating weight sets, 436 MB, which L2 cannot hold); the
cold number is the one that describes a 40-layer decode step.

Weight dtype
------------
bf16 and f32 are both accepted, by *building a different kernel*
(`functools.lru_cache` keyed on the dtype) rather than by converting a 33.5 MB
tensor per call. The bf16 path is the product path and uses `T.gemm`. The f32
path exists so the comparison against the authored reference is tight, and it is
deliberately *not* `T.gemm`: tilelang's f32 `T.gemm` runs on TF32 (measured
8.0e-4 relative error, which is looser than the bf16 path), so the f32 path is a
plain f32 accumulation over the same staged tile -- 4x slower, 7e-7 accurate.
"""
from __future__ import annotations

import functools

import tilelang
import tilelang.language as T
import torch

try:  # `import kernels.attn` / `python -m kernels.attn`
    from .basic import BM
except ImportError:  # `python kernels/attn.py`
    from basic import BM

#: config.REAL, spelled out. Published dimensions, never derived: `head_dim` is
#: 256 while `hidden / n_q_heads` is 128.
_H = 2048
_HQ = 16
_HKV = 2
_G = _HQ // _HKV
_D = 256
_ROT = 64
_HALF = _ROT // 2
_ROWS = 4096
_EPS = 1e-6

_QG = _HQ * _D * 2  # 8192 -- q and gate interleaved *inside* each head
_KV = _HKV * _D  # 512
_OI = _HQ * _D  # 4096 -- o_proj's contraction

#: The identity of the log-sum-exp semiring: a partial with `l = 0` and this
#: `m` contributes nothing to the merge. Not `-inf`, because `exp(-inf - -inf)`
#: is NaN and a split that saw no position at all has exactly that `m`.
_MNEG = -1.0e30

#: (BN, BK, threads, stages, K-splits) per weight dtype, from a sweep over the
#: real shapes with the weights rotated so L2 cannot hold them.
#:
#: One block reads at most ~20 GB/s no matter how deep its pipeline (measured:
#: 8 blocks on the 4.2 MB k/v pair take 30 us, 128 blocks take 4.1 us), so the
#: only thing that sets the time is how many blocks are resident and how evenly
#: the bytes are spread over them. 132 SMs and one block each: 64 blocks
#: measured 11.3 us on w_qg, 128 blocks 10.6, 256 blocks 13.4. K-splits are how
#: w_qg reaches 128 blocks while BN stays >= 128 -- a 128-column bf16 tile row
#: is 256 B, one full sector pair, where BN=32 would fetch 64 B rows.
#:
#: The same measurement is why w_k and w_v are *not* extra blocks in the same
#: grid: at BN=128 they are only 16 tiles, those 16 blocks become the critical
#: path at 20 GB/s each, and the launch goes from 11.0 us to 21.9. They are
#: instead one extra 128x128 tile bolted onto each of the 128 w_qg blocks.
_CFG_PROJ = {"bfloat16": (128, 64, 128, 4, 2), "float32": (128, 64, 128, 3, 2)}
_CFG_OUT = {"bfloat16": (128, 64, 128, 4, 8), "float32": (128, 64, 128, 3, 8)}

#: Rows of w_k / w_v one block contracts in its bolted-on tile. Fixed by the
#: arithmetic in `_proj_kernel`: (2048 / _KVK) * (512 / BN) tiles per matrix,
#: two matrices, must equal the w_qg block count.
_KVK = 128

#: Context positions per attention tile, and the cap on how many blocks one kv
#: head's context is split across. P=32 beat P=64 at every C (the tile's
#: global->shared staging is the fixed cost and it scales with P).
#:
#: The split cap is a straight trade, measured at C=2047: 64 splits gives
#: kernel 2 128 blocks (11.0 us) and kernel 3 65 log-sum-exp slots to re-merge
#: in every o_proj block (13.9 us); 32 splits gives 64 blocks (15.1 us) and 33
#: slots (8.9 us). 24.0 us against 24.9 -- a wash at the long end, and 32 is
#: strictly better at every shorter context, where kernel 2 is not split-bound
#: but kernel 3 still pays per slot.
_ATT_P = 32
_ATT_SPLITS = 32


def _wdt(w: torch.Tensor) -> str:
    """The kernel dtype name for a weight tensor. Also the lru_cache key."""
    if w.dtype is torch.bfloat16:
        return "bfloat16"
    if w.dtype is torch.float32:
        return "float32"
    raise TypeError(f"weights must be bf16 or f32, got {w.dtype}")


# --------------------------------------------------------------------------
# Expression helpers. These live at module level on purpose: tilelang's eager
# builder rewrites the *decorated* function's AST, so a plain Python function
# called from inside a kernel body is ordinary Python building PrimExprs -- no
# SSA rebinding rules, no `T.alloc_var` needed for the accumulator.
# --------------------------------------------------------------------------
def _psum(buf, idx, splits: int):
    """A split-K kernel's `splits` partial rows, summed at flat column `idx`."""
    acc = buf[0, idx]
    for t in range(1, splits):
        acc = acc + buf[t, idx]
    return acc


def _rope(val, partner, cos, sin, d):
    """HF `rotate_half` RoPE at one output index `d`, verified against
    `modeling_qwen3_5_moe.rotate_half` / `apply_rotary_pos_emb`:

        x1, x2 = x[:32], x[32:64];  rotate_half = cat(-x2, x1)
        y[0:64]   = x[0:64] * cos + rotate_half(x[0:64]) * sin
        y[64:256] = x[64:256]                              (pass_dim untouched)

    which per index is `y[i] = x[i]*cos[i] - x[i+32]*sin[i]` for i < 32,
    `x[i]*cos[i] + x[i-32]*sin[i]` for 32 <= i < 64, and `x[i]` above that.
    `cos` is 64 wide and already `cat(freqs, freqs)`, so `cos[i] == cos[i+32]`
    and indexing it at `i` (as HF does) is the same as indexing at `i % 32`.
    """
    return T.if_then_else(
        d < _HALF,
        val * cos - partner * sin,
        T.if_then_else(d < _ROT, val * cos + partner * sin, val),
    )


def _partner(d):
    """The index `rotate_half` pairs `d` with. For `d >= _ROT` the value is
    unused but must still be in bounds, and `d - _HALF` is."""
    return T.if_then_else(d < _HALF, d + _HALF, d - _HALF)


# --------------------------------------------------------------------------
# 0. partial_rope / partial_rope_kv -- the two standalone entry points
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _rope_kernel(HEADS: int):
    """`y = rope(x)` over `HEADS x 256`, rotating only the leading 64.

    One block: 4096 elements is a rounding error next to a launch. The two head
    counts get two kernels because `HEADS` is a shape.
    """

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((HEADS, _D), "float32"),
            Cos: T.Tensor((_ROWS, _ROT), "float32"),
            Sin: T.Tensor((_ROWS, _ROT), "float32"),
            Pos: T.Tensor((1,), "int32"),
            Y: T.Tensor((HEADS, _D), "float32"),
        ):
            with T.Kernel(1, threads=256) as _:
                # The position is read on the device: a decode step must not
                # need the previous step's sampled token back on the host.
                p = Pos[0]
                for h, d in T.Parallel(HEADS, _D):
                    cd = T.min(d, _ROT - 1)  # clamped: if_then_else is not lazy
                    Y[h, d] = _rope(X[h, d], X[h, _partner(d)], Cos[p, cd], Sin[p, cd], d)

        return main

    return build()


def partial_rope(x, cos_cache, sin_cache, pos_ids):
    """(1,1,16,256) f32 -> (1,1,16,256) f32."""
    y = torch.empty_like(x)
    _rope_kernel(_HQ)(x.reshape(_HQ, _D), cos_cache, sin_cache, pos_ids, y.view(_HQ, _D))
    return y


def partial_rope_kv(x, cos_cache, sin_cache, pos_ids):
    """(1,1,2,256) f32 -> (1,1,2,256) f32."""
    y = torch.empty_like(x)
    _rope_kernel(_HKV)(x.reshape(_HKV, _D), cos_cache, sin_cache, pos_ids, y.view(_HKV, _D))
    return y


# --------------------------------------------------------------------------
# 1. rms_norm(hidden) + the q||gate, k and v projections
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _proj_kernel(WDT: str):
    """One launch for `rms_norm(hidden) @ {w_qg, w_k, w_v}`.

    The grid is w_qg's tiling and nothing else: 64 column tiles x 2 K-splits =
    128 blocks, one per SM. w_k and w_v are 4.2 MB between them, which is 1/8th
    of w_qg -- as blocks of their own they would be 16 tiles running at 20 GB/s
    and would double the launch, so instead each block carries **one extra
    128x128 tile** of one of them. 128 blocks, 128 such tiles (16 row-tiles x 4
    column-tiles x 2 matrices), exact fit; +12% traffic per block.

    Outputs are (splits, N) partial rows -- 2 for q||gate, 16 for k and v; the
    two consumers sum them in their own prologues, which is a few thousand adds
    against this kernel's 37.7 MB.
    """
    BN, BK, TH, ST, KS = _CFG_PROJ[WDT]
    F32 = WDT == "float32"
    KB = _H // KS  # rows of w_qg this block contracts
    KO = KB // BK
    NBQ = _QG // BN  # 64 column tiles of w_qg
    NB = NBQ * KS  # ... and the whole grid
    NKR = _H // _KVK  # 16 row-tiles of w_k / w_v -> KR/VR partial rows
    NKN = _KV // BN  # 4 column tiles
    assert NKR * NKN * 2 == NB, "the k/v tiling must cover the w_qg grid exactly"

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((_H,), "float32"),
            Gin: T.Tensor((_H,), "float32"),
            Wqg: T.Tensor((_H, _QG), WDT),
            Wk: T.Tensor((_H, _KV), WDT),
            Wv: T.Tensor((_H, _KV), WDT),
            QG: T.Tensor((KS, _QG), "float32"),
            KR: T.Tensor((NKR, _KV), "float32"),
            VR: T.Tensor((NKR, _KV), "float32"),
        ):
            with T.Kernel(NB, threads=TH) as b:
                # `KS` and `NKR` are spelled out here because they have to be:
                # tilelang evaluates the (stringified) parameter annotations
                # against this function's *closure cells*, and Python only
                # creates a cell for a name the nested function's code
                # references. A dimension that appears only in an annotation
                # raises `NameError` -- basic.py's `embed_row` records the same
                # trap. `nkr` doubles as a bounds clamp that is always a no-op.
                kb = _H // KS
                nb = b % NBQ
                ks = b // NBQ
                ws = T.alloc_shared((BK, BN), WDT)
                ws2 = T.alloc_shared((_KVK, BN), WDT)
                sq = T.alloc_fragment((_H,), "float32")
                tot = T.alloc_fragment((1,), "float32")
                rsc = T.alloc_shared((1,), "float32")
                xn = T.alloc_shared((_H,), "float32")

                # `tf.rms_norm` is `x * rsqrt(mean(x^2) + eps) * gamma` flat --
                # no `1 +`; model.py's converters folded that into the gammas.
                # Recomputed in all 128 blocks: 16 KB of reads each, against
                # 294 KB of weights each, and it saves a launch.
                for i in T.Parallel(_H):
                    sq[i] = X[i] * X[i]
                T.reduce_sum(sq, tot, dim=0)
                if T.get_thread_binding() == 0:
                    rsc[0] = T.rsqrt(tot[0] / T.cast(_H, "float32") + _EPS)
                T.sync_threads()
                # Whole normalised vector in shared: the k/v tile needs rows
                # this block's own K-split does not cover.
                for i in T.Parallel(_H):
                    xn[i] = X[i] * rsc[0] * Gin[i]
                T.sync_threads()

                col = nb * BN
                # k/v tile: block b owns tile b, w_k for b < 64 and w_v after.
                iskv = b % (NB // 2)
                kt = T.min(iskv // NKN, NKR - 1)
                nt = (iskv % NKN) * BN
                if F32:
                    av = T.alloc_fragment((BN,), "float32")
                    av2 = T.alloc_fragment((BN,), "float32")
                    T.clear(av)
                    T.clear(av2)
                else:
                    xs = T.alloc_shared((BM, KB), "bfloat16")
                    xs2 = T.alloc_shared((BM, _KVK), "bfloat16")
                    acc = T.alloc_fragment((BM, BN), "float32")
                    acc2 = T.alloc_fragment((BM, BN), "float32")
                    osh = T.alloc_shared((BM, BN), "float32")
                    osh2 = T.alloc_shared((BM, BN), "float32")
                    for j in T.Parallel(KB):
                        v = xn[ks * kb + j]
                        hi = T.cast(v, "bfloat16")
                        xs[0, j] = hi  # rows 2..15 stay uninitialised, by design
                        xs[1, j] = T.cast(v - T.cast(hi, "float32"), "bfloat16")
                    for j in T.Parallel(_KVK):
                        v = xn[kt * _KVK + j]
                        hi = T.cast(v, "bfloat16")
                        xs2[0, j] = hi
                        xs2[1, j] = T.cast(v - T.cast(hi, "float32"), "bfloat16")
                    T.clear(acc)
                    T.clear(acc2)

                # The k/v tile is issued first and consumed last, so its copy
                # overlaps the w_qg pipeline instead of sitting exposed. It is
                # its own shared tile: two `T.Pipelined` loops each multi-buffer
                # the buffer they copy into, and sharing one is rejected (see
                # the module docstring), so this one is a plain copy.
                if b < NB // 2:
                    T.copy(Wk[kt * _KVK : (kt + 1) * _KVK, nt : nt + BN], ws2)
                else:
                    T.copy(Wv[kt * _KVK : (kt + 1) * _KVK, nt : nt + BN], ws2)
                T.sync_threads()

                for ko in T.Pipelined(KO, num_stages=ST):
                    r0 = ks * kb + ko * BK
                    T.copy(Wqg[r0 : r0 + BK, col : col + BN], ws)
                    if F32:
                        # Deliberately not T.gemm: its f32 path is TF32, which
                        # is *less* accurate than the bf16 hi/lo path.
                        for j in T.Parallel(BN):
                            for kk in T.serial(BK):
                                av[j] += xn[r0 + kk] * ws[kk, j]
                    else:
                        T.gemm(xs[:, ko * BK : (ko + 1) * BK], ws, acc)

                if F32:
                    for j in T.Parallel(BN):
                        for kk in T.serial(_KVK):
                            av2[j] += xn[kt * _KVK + kk] * ws2[kk, j]
                else:
                    T.gemm(xs2, ws2, acc2)
                    # `acc` is a fragment whose layout spreads both axes over
                    # threads, so `acc[0, j]` in a 1-D T.Parallel is rejected as
                    # "Loop layout is not injective". Row 0 goes through shared.
                    T.copy(acc, osh)
                    T.copy(acc2, osh2)
                for j in T.Parallel(BN):
                    QG[ks, col + j] = av[j] if F32 else osh[0, j] + osh[1, j]
                    v2 = av2[j] if F32 else osh2[0, j] + osh2[1, j]
                    if b < NB // 2:
                        KR[kt, nt + j] = v2
                    else:
                        VR[kt, nt + j] = v2

        return main

    return build(), NKR


# --------------------------------------------------------------------------
# 2. the attention itself
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _attn_kernel(CAP: int, NCTX: int | None, KSQ: int, KSKV: int, WCACHE: bool):
    """Softmax over `NCTX` cached positions plus this token, per query head.

    One block per (kv head, context split). The group is the unit, not the
    head: all eight query heads of a GQA group read the same K and V, so a
    block that owns the group stages each cache tile once and scores it eight
    times. That is 8x less traffic than one block per query head, and it makes
    the score a 16x256 by 256xP MMA -- eight q rows and their eight low halves.

    `NCTX` is either a Python int (`full_attention`: the cache length is a
    shape) or None, meaning "read it from `Pos[0]`" (`full_attention_fixed`:
    the cache has a fixed capacity and a device-side fill level). The body is
    the same either way; a constant just folds the guards away.

    This token's own position is *not* appended to the tile stream. It is its
    own log-sum-exp partial, in slot `NSPLIT`, and a cheap one: `exp(s - s) = 1`,
    so that partial is `(m = score, l = 1, o = v)` with no exponential at all.
    That is also what keeps the fixed-capacity variant race-free -- the slot it
    writes into the cache is never read by this step.

    Softmax is an ordinary numerically-stable one-pass over the C+1 positions,
    split across blocks and merged by log-sum-exp in kernel 3. The reference
    merges two groups (cache, new token) instead; the two are algebraically the
    same thing, and this one also handles C = 0, where every context block
    finds its range empty, leaves `(m = -1e30, l = 0, o = 0)`, and the merge
    drops it. Nothing divides by zero: the new-token slot always has `l = 1`.
    """
    P = _ATT_P
    NT = max((CAP + P - 1) // P, 1)  # context tiles the kernel may look at
    TPB = (NT + min(_ATT_SPLITS, NT) - 1) // min(_ATT_SPLITS, NT)
    NSPLIT = (NT + TPB - 1) // TPB  # recomputed from TPB: no empty blocks
    SLOTS = NSPLIT + 1
    CDIM = max(CAP, 1)  # a 0-length tensor is not a legal kernel argument
    TH = 128

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            QG: T.Tensor((KSQ, _QG), "float32"),
            KR: T.Tensor((KSKV, _KV), "float32"),
            VR: T.Tensor((KSKV, _KV), "float32"),
            Gq: T.Tensor((_D,), "float32"),
            Gk: T.Tensor((_D,), "float32"),
            Cos: T.Tensor((_ROWS, _ROT), "float32"),
            Sin: T.Tensor((_ROWS, _ROT), "float32"),
            Pos: T.Tensor((1,), "int32"),
            Kc: T.Tensor((CDIM, _HKV, _D), "float32"),
            Vc: T.Tensor((CDIM, _HKV, _D), "float32"),
            Scale: T.Tensor((1,), "float32"),
            Op: T.Tensor((SLOTS, _HQ, _D), "float32"),
            Mp: T.Tensor((SLOTS, _HQ), "float32"),
            Lp: T.Tensor((SLOTS, _HQ), "float32"),
            Kro: T.Tensor((_HKV, _D), "float32"),
            Vo: T.Tensor((_HKV, _D), "float32"),
            Oacc: T.Tensor((_H,), "float32"),
        ):
            with T.Kernel(_HKV * NSPLIT, threads=TH) as b:
                g = b % _HKV
                s = b // _HKV
                tid = T.get_thread_binding()
                pos = Pos[0]
                # Valid cached slots. A constant folds every guard below.
                nctx = pos if NCTX is None else NCTX

                qsh = T.alloc_shared((_G, _D), "float32")
                qs = T.alloc_shared((2 * _G, _D), "bfloat16")
                knsh = T.alloc_shared((_D,), "float32")
                hf = T.alloc_fragment((_G, _D), "float32")
                hr = T.alloc_fragment((_G,), "float32")
                hs = T.alloc_shared((_G,), "float32")
                df = T.alloc_fragment((_D,), "float32")
                d1 = T.alloc_fragment((1,), "float32")
                ds = T.alloc_shared((1,), "float32")
                # Everything the two head norms and the RoPE touch, staged once
                # in shared. Not an optimisation of taste: the RoPE reads index
                # `d` and index `d +- 32`, so straight off global it is a
                # data-dependent scalar address, unvectorisable, and at C = 0
                # this kernel has *two* blocks on the whole GPU -- nothing to
                # hide the latency behind. Measured 6.8 us -> 2.4 us.
                qraw = T.alloc_shared((_G, _D), "float32")
                kraw = T.alloc_shared((_D,), "float32")
                vraw = T.alloc_shared((_D,), "float32")
                gqs = T.alloc_shared((_D,), "float32")
                gks = T.alloc_shared((_D,), "float32")
                css = T.alloc_shared((_ROT,), "float32")
                sss = T.alloc_shared((_ROT,), "float32")

                kf = T.alloc_shared((P, _D), "float32")
                vf = T.alloc_shared((P, _D), "float32")
                kh = T.alloc_shared((P, _D), "bfloat16")
                kl = T.alloc_shared((P, _D), "bfloat16")
                vh = T.alloc_shared((P, _D), "bfloat16")
                vl = T.alloc_shared((P, _D), "bfloat16")
                accs = T.alloc_fragment((2 * _G, P), "float32")
                ssh = T.alloc_shared((2 * _G, P), "float32")
                scf = T.alloc_fragment((_G, P), "float32")
                red = T.alloc_fragment((_G,), "float32")
                nsh = T.alloc_shared((_G,), "float32")
                psh = T.alloc_shared((2 * _G, P), "bfloat16")
                acco = T.alloc_fragment((2 * _G, _D), "float32")
                osh = T.alloc_shared((2 * _G, _D), "float32")
                msh = T.alloc_shared((_G,), "float32")
                lsh = T.alloc_shared((_G,), "float32")
                csh = T.alloc_shared((_G,), "float32")

                # ---- stage, then head-norm over the 256 axis, then RoPE ----
                qbase = g * _G * 2 * _D  # q||gate is head-major, 512 per head
                for j, d in T.Parallel(_G, _D):
                    v = _psum(QG, qbase + j * 2 * _D + d, KSQ)
                    qraw[j, d] = v
                    hf[j, d] = v * v
                for d in T.Parallel(_D):
                    v = _psum(KR, g * _D + d, KSKV)
                    kraw[d] = v
                    df[d] = v * v
                    vraw[d] = _psum(VR, g * _D + d, KSKV)
                    gqs[d] = Gq[d]
                    gks[d] = Gk[d]
                    if d < _ROT:
                        css[d] = Cos[pos, d]
                        sss[d] = Sin[pos, d]
                T.reduce_sum(hf, hr, dim=1)
                T.reduce_sum(df, d1, dim=0)
                for j in T.Parallel(_G):
                    hr[j] = T.rsqrt(hr[j] / T.cast(_D, "float32") + _EPS)
                T.copy(hr, hs)
                if tid == 0:
                    ds[0] = T.rsqrt(d1[0] / T.cast(_D, "float32") + _EPS)
                T.sync_threads()
                sc = Scale[0]
                for j, d in T.Parallel(_G, _D):
                    dp = _partner(d)
                    cd = T.min(d, _ROT - 1)  # clamped: if_then_else is not lazy
                    xr = qraw[j, d] * hs[j] * gqs[d]
                    xp = qraw[j, dp] * hs[j] * gqs[dp]
                    # The scale is folded into q once, here, instead of into
                    # every score.
                    y = _rope(xr, xp, css[cd], sss[cd], d) * sc
                    qsh[j, d] = y
                    hi = T.cast(y, "bfloat16")
                    qs[j, d] = hi
                    qs[_G + j, d] = T.cast(y - T.cast(hi, "float32"), "bfloat16")
                for d in T.Parallel(_D):
                    dp = _partner(d)
                    cd = T.min(d, _ROT - 1)
                    knsh[d] = _rope(kraw[d] * ds[0] * gks[d],
                                    kraw[dp] * ds[0] * gks[dp], css[cd], sss[cd], d)
                if tid < _G:
                    msh[tid] = _MNEG
                    lsh[tid] = 0.0
                T.clear(acco)
                T.sync_threads()

                if s == 0:
                    for d in T.Parallel(_D):
                        Kro[g, d] = knsh[d]
                        Vo[g, d] = vraw[d]
                    if WCACHE:
                        # A deliberate departure from the reference's "a step
                        # must not mutate a tensor it was given": a CUDA graph
                        # replays fixed addresses, so a growing cache cannot be
                        # re-pointed between steps and the step has to write
                        # into the capacity it was handed. Slot `pos` is not
                        # read by this step (nctx == pos), so no block races.
                        # `pos < CDIM` is a memory-safety guard, not a semantic
                        # one: a step past the capacity it was given has no
                        # defined answer, but it must not write off the end.
                        for d in T.Parallel(_D):
                            if pos < CDIM:
                                Kc[pos, g, d] = knsh[d]
                                Vc[pos, g, d] = vraw[d]
                    # This token's own position, as a log-sum-exp partial.
                    for j, d in T.Parallel(_G, _D):
                        hf[j, d] = qsh[j, d] * knsh[d]
                    T.reduce_sum(hf, hr, dim=1)
                    T.copy(hr, hs)
                    T.sync_threads()
                    for j in T.Parallel(_G):
                        Mp[SLOTS - 1, g * _G + j] = hs[j]
                        Lp[SLOTS - 1, g * _G + j] = 1.0
                    for j, d in T.Parallel(_G, _D):
                        Op[SLOTS - 1, g * _G + j, d] = vraw[d]
                if b == 0:
                    # kernel 3 accumulates o_proj's K-splits with atomics, so
                    # somebody has to zero the accumulator. Doing it here costs
                    # 2048 stores in one block and saves a launch.
                    for i in T.Parallel(_H):
                        Oacc[i] = 0.0

                # ---- the cached positions ----------------------------------
                for it in T.serial(TPB):
                    base = (s * TPB + it) * P
                    if base < nctx:
                        if base + P <= nctx:
                            # Whole tile in range: one strided async copy each.
                            T.copy(Kc[base : base + P, g, :], kf)
                            T.copy(Vc[base : base + P, g, :], vf)
                        else:
                            # Only the tail tile takes the scalar path. Slots
                            # past `nctx` are zeroed rather than left alone:
                            # the fixed-capacity cache may hold NaN there, and
                            # `0 * NaN` in the PV product would spread it even
                            # though the score is masked.
                            for pp, d in T.Parallel(P, _D):
                                t = base + pp
                                tc = T.min(t, CDIM - 1)
                                kf[pp, d] = T.if_then_else(
                                    t < nctx, Kc[tc, g, d], T.float32(0.0))
                                vf[pp, d] = T.if_then_else(
                                    t < nctx, Vc[tc, g, d], T.float32(0.0))
                        T.sync_threads()
                        for pp, d in T.Parallel(P, _D):
                            hik = T.cast(kf[pp, d], "bfloat16")
                            kh[pp, d] = hik
                            kl[pp, d] = T.cast(kf[pp, d] - T.cast(hik, "float32"), "bfloat16")
                            hiv = T.cast(vf[pp, d], "bfloat16")
                            vh[pp, d] = hiv
                            vl[pp, d] = T.cast(vf[pp, d] - T.cast(hiv, "float32"), "bfloat16")
                        T.sync_threads()

                        T.clear(accs)
                        T.gemm(qs, kh, accs, transpose_B=True)
                        T.gemm(qs, kl, accs, transpose_B=True)
                        T.copy(accs, ssh)
                        # rows 0..7 hold q_hi.k, rows 8..15 q_lo.k; their sum is
                        # the f32-accurate score. Writes land in rows < 8 and
                        # reads come from rows >= 8, so this is race-free.
                        for j, pp in T.Parallel(_G, P):
                            t = base + pp
                            scf[j, pp] = T.if_then_else(
                                t < nctx, ssh[j, pp] + ssh[_G + j, pp], T.float32(_MNEG))
                        # The tile's max and sum are fragment reduces, not a
                        # serial loop on eight threads: the serial form is a
                        # P-deep dependent chain of shared loads, twice per
                        # tile, and it measured ~1.2 us of the tile's ~5.
                        T.reduce_max(scf, red, dim=1)
                        for j in T.Parallel(_G):
                            red[j] = T.max(red[j], msh[j])
                        T.copy(red, nsh)
                        T.sync_threads()
                        if tid < _G:
                            csh[tid] = T.exp(msh[tid] - nsh[tid])
                            msh[tid] = nsh[tid]
                        T.sync_threads()
                        for j, pp in T.Parallel(_G, P):
                            e = T.exp(scf[j, pp] - msh[j])
                            hie = T.cast(e, "bfloat16")
                            psh[j, pp] = hie
                            psh[_G + j, pp] = T.cast(e - T.cast(hie, "float32"), "bfloat16")
                            scf[j, pp] = e
                        T.reduce_sum(scf, red, dim=1)
                        T.copy(red, nsh)
                        T.sync_threads()
                        if tid < _G:
                            lsh[tid] = lsh[tid] * csh[tid] + nsh[tid]
                        for j, d in T.Parallel(2 * _G, _D):
                            acco[j, d] = acco[j, d] * csh[j % _G]
                        T.sync_threads()
                        T.gemm(psh, vh, acco)
                        T.gemm(psh, vl, acco)

                T.copy(acco, osh)
                for j, d in T.Parallel(_G, _D):
                    Op[s, g * _G + j, d] = osh[j, d] + osh[_G + j, d]
                if tid < _G:
                    Mp[s, g * _G + tid] = msh[tid]
                    Lp[s, g * _G + tid] = lsh[tid]

        return main

    return build(), NSPLIT, SLOTS


# --------------------------------------------------------------------------
# 3. log-sum-exp merge + output gate + o_proj
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _out_kernel(WDT: str, SLOTS: int, KSQ: int):
    """`out = (merge(partials) * sigmoid(gate)) @ w_o`.

    o_proj contracts 4096, which at BN=128 is only 16 column tiles -- a fifth of
    the machine. So K is split 8 ways and the 128 blocks accumulate into the
    output with `T.atomic_add` (kernel 2 zeroed it). Summation order is then not
    reproducible bit-for-bit; at 8 f32 addends that is ~1e-7 and buys 2x.

    Each K-split owns 512 of the 4096, which is exactly two query heads, so a
    block merges and gates only the two heads it contracts.
    """
    BN, BK, TH, ST, KS = _CFG_OUT[WDT]
    F32 = WDT == "float32"
    KB = _OI // KS
    KO = KB // BK
    HPB = KB // _D  # query heads per K-split
    NBO = _H // BN

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Op: T.Tensor((SLOTS, _HQ, _D), "float32"),
            Mp: T.Tensor((SLOTS, _HQ), "float32"),
            Lp: T.Tensor((SLOTS, _HQ), "float32"),
            QG: T.Tensor((KSQ, _QG), "float32"),
            Wo: T.Tensor((_OI, _H), WDT),
            Out: T.Tensor((_H,), "float32"),
        ):
            with T.Kernel(NBO * KS, threads=TH) as b:
                nb = b % NBO
                ks = b // NBO
                h0 = ks * HPB
                tid = T.get_thread_binding()
                ws = T.alloc_shared((BK, BN), WDT)
                wgt = T.alloc_shared((SLOTS, HPB), "float32")
                lsh = T.alloc_shared((HPB,), "float32")
                num = T.alloc_fragment((HPB, _D), "float32")

                # Log-sum-exp merge of the context splits against their joint
                # max. A split that saw no position has l = 0 and m = -1e30, so
                # it contributes exp(-1e30 - M) * 0 = 0 and never a NaN.
                if tid < HPB:
                    mx = T.alloc_var("float32", init=_MNEG)
                    for t in T.serial(SLOTS):
                        mx = T.max(mx, Mp[t, h0 + tid])
                    den = T.alloc_var("float32", init=0.0)
                    for t in T.serial(SLOTS):
                        w = T.exp(Mp[t, h0 + tid] - mx)
                        wgt[t, tid] = w
                        den = den + Lp[t, h0 + tid] * w
                    lsh[tid] = den
                T.clear(num)
                T.sync_threads()
                # Serial over the slots *inside* the parallel loop: the other
                # way round is 65 sequential rounds of a 512-element global
                # gather, each paying full latency, and it measured 6 us. This
                # way each thread has SLOTS independent loads in flight.
                for j, d in T.Parallel(HPB, _D):
                    for t in T.serial(SLOTS):
                        num[j, d] += Op[t, h0 + j, d] * wgt[t, j]

                if F32:
                    xf = T.alloc_shared((KB,), "float32")
                    av = T.alloc_fragment((BN,), "float32")
                    T.clear(av)
                else:
                    xs = T.alloc_shared((BM, KB), "bfloat16")
                    acc = T.alloc_fragment((BM, BN), "float32")
                    osh = T.alloc_shared((BM, BN), "float32")
                    T.clear(acc)
                for j, d in T.Parallel(HPB, _D):
                    # Head-major flattening on both sides: gate entry (h, d)
                    # meets attention entry (h, d). The gate lives beside the
                    # query inside each head, at offset 256 of that head's 512.
                    gate = _psum(QG, (h0 + j) * 2 * _D + _D + d, KSQ)
                    v = (num[j, d] / lsh[j]) / (1.0 + T.exp(-gate))
                    if F32:
                        xf[j * _D + d] = v
                    else:
                        hi = T.cast(v, "bfloat16")
                        xs[0, j * _D + d] = hi
                        xs[1, j * _D + d] = T.cast(v - T.cast(hi, "float32"), "bfloat16")
                T.sync_threads()

                for ko in T.Pipelined(KO, num_stages=ST):
                    r0 = ks * KB + ko * BK
                    T.copy(Wo[r0 : r0 + BK, nb * BN : (nb + 1) * BN], ws)
                    if F32:
                        for j in T.Parallel(BN):
                            for kk in T.serial(BK):
                                av[j] += xf[ko * BK + kk] * ws[kk, j]
                    else:
                        T.gemm(xs[:, ko * BK : (ko + 1) * BK], ws, acc)
                if not F32:
                    T.copy(acc, osh)
                for j in T.Parallel(BN):
                    T.atomic_add(Out[nb * BN + j], av[j] if F32 else osh[0, j] + osh[1, j])

        return main

    return build()


# --------------------------------------------------------------------------
# the two fused entry points
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _empty_kv(device: str):
    """A stand-in for a zero-length cache. `T.Tensor((0, 2, 256))` is not a
    legal kernel argument, and `C = 0` folds every read of it away, so what is
    passed only has to exist. Cached, so it is a fixed address a graph can
    capture."""
    return torch.zeros(1, _HKV, _D, dtype=torch.float32, device=device)


def _run(hidden, gamma_in, w_qg, w_k, w_v, gamma_q, gamma_k, cos_cache, sin_cache,
         pos, k_cache, v_cache, scale, w_o, *, nctx, cap, wcache, out):
    """The three launches. Shared by both entry points; they differ only in
    whether the cache length is a shape (`nctx = C`) or a device value
    (`nctx = None`), and in whether the cache is written."""
    wdt = _wdt(w_qg)
    dev = hidden.device
    ksq = _CFG_PROJ[wdt][4]
    kern1, kskv = _proj_kernel(wdt)

    qg = torch.empty(ksq, _QG, dtype=torch.float32, device=dev)
    kr = torch.empty(kskv, _KV, dtype=torch.float32, device=dev)
    vr = torch.empty(kskv, _KV, dtype=torch.float32, device=dev)
    kern1(
        hidden.reshape(_H), gamma_in,
        w_qg.reshape(_H, _QG), w_k.reshape(_H, _KV), w_v.reshape(_H, _KV),
        qg, kr, vr,
    )

    kern2, _, slots = _attn_kernel(cap, nctx, ksq, kskv, wcache)
    op = torch.empty(slots, _HQ, _D, dtype=torch.float32, device=dev)
    mp = torch.empty(slots, _HQ, dtype=torch.float32, device=dev)
    lp = torch.empty(slots, _HQ, dtype=torch.float32, device=dev)
    k_rope = torch.empty(_HKV, _D, dtype=torch.float32, device=dev)
    v_new = torch.empty(_HKV, _D, dtype=torch.float32, device=dev)
    if cap == 0:
        kc = vc = _empty_kv(str(dev))
    else:
        # `.view`, not `.reshape`: the fixed-capacity variant writes this token's
        # k/v through it, and a reshape that silently copied would drop them.
        kc = k_cache.view(cap, _HKV, _D)
        vc = v_cache.view(cap, _HKV, _D)
    kern2(qg, kr, vr, gamma_q, gamma_k, cos_cache, sin_cache, pos, kc, vc,
          scale.reshape(1), op, mp, lp, k_rope, v_new, out)

    _out_kernel(wdt, slots, ksq)(op, mp, lp, qg, w_o.reshape(_OI, _H), out)
    return k_rope, v_new


def full_attention(hidden, gamma_in, w_qg, w_k, w_v, gamma_q, gamma_k,
                   cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o):
    """The reference contract: the cache length is a shape, and this token's
    key/value come back for the caller to append.

    Returns `(out (1,1,2048), k_rope (1,1,2,256), v (1,1,2,256))`.
    """
    C = int(k_cache.shape[1])  # a host-side shape, not a device value
    out = torch.empty(_H, dtype=torch.float32, device=hidden.device)
    k_rope, v_new = _run(
        hidden, gamma_in, w_qg, w_k, w_v, gamma_q, gamma_k, cos_cache, sin_cache,
        pos_ids, k_cache, v_cache, scale, w_o,
        nctx=C, cap=C, wcache=False, out=out,
    )
    return (out.view(1, 1, _H),
            k_rope.view(1, 1, _HKV, _D),
            v_new.view(1, 1, _HKV, _D))


def full_attention_fixed(hidden, gamma_in, w_qg, w_k, w_v, gamma_q, gamma_k,
                         cos_cache, sin_cache, k_cache, v_cache, pos, scale, w_o, out):
    """The CUDA-graph contract: fixed-capacity cache, device-side fill level.

    `pos` is a device tensor holding this token's absolute position, which is
    also the number of already-valid slots. Nothing about it is read on the
    host, so one capture serves every step: `pos.fill_(n)` between replays is
    enough. Slots `> pos` are stale -- they are excluded by a `t < pos`
    predicate on the score, not by zeroing, and they are also zeroed on the way
    into shared memory so a NaN slot cannot reach the PV product.

    Writes `out` in place, and appends this token's k_rope / v to
    `k_cache[0, pos]` / `v_cache[0, pos]`. Returns None.
    """
    cap = int(k_cache.shape[1])
    _run(hidden, gamma_in, w_qg, w_k, w_v, gamma_q, gamma_k, cos_cache, sin_cache,
         pos, k_cache, v_cache, scale, w_o,
         nctx=None, cap=cap, wcache=True, out=out.view(_H))


__all__ = ["full_attention", "full_attention_fixed", "partial_rope", "partial_rope_kv"]


# ==========================================================================
# self-test
# ==========================================================================
if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    DEV = "cuda"

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

    def relerr(got, ref):
        got = got.double()
        ref = ref.double()
        return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()

    # ---- plain-torch twins, written out here rather than imported --------
    def t_rope(x, cos, sin, pos_ids):
        p = int(pos_ids[0])
        r, tail = x[..., :_ROT], x[..., _ROT:]
        half = _ROT // 2
        rot = torch.cat((-r[..., half:], r[..., :half]), -1)  # HF rotate_half
        return torch.cat((r * cos[p] + rot * sin[p], tail), -1)

    def t_rms(x, gm):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + _EPS) * gm

    def t_attn(hidden, gamma_in, w_qg, w_k, w_v, gamma_q, gamma_k, cos, sin,
               pos_ids, k_cache, v_cache, scale, w_o):
        hn = t_rms(hidden, gamma_in)
        qg = (hn @ w_qg[0].float()).view(1, 1, _HQ, 2 * _D)
        q = t_rope(t_rms(qg[..., :_D], gamma_q), cos, sin, pos_ids)
        gate = qg[..., _D:]
        k = t_rope(t_rms((hn @ w_k[0].float()).view(1, 1, _HKV, _D), gamma_k),
                   cos, sin, pos_ids)
        v = (hn @ w_v[0].float()).view(1, 1, _HKV, _D)
        kall = torch.cat([k_cache.float(), k], 1)[0]  # (C+1, HKV, D)
        vall = torch.cat([v_cache.float(), v], 1)[0]
        attn = torch.empty(_HQ, _D, dtype=torch.float64, device=hidden.device)
        for h in range(_HQ):
            sc = (q[0, 0, h].double() * scale.reshape(()).double()) @ kall[:, h // _G].double().T
            pr = torch.softmax(sc, -1)
            attn[h] = (pr[:, None] * vall[:, h // _G].double()).sum(0)
        gated = attn.reshape(_OI) * torch.sigmoid(gate.reshape(_OI)).double()
        return (gated @ w_o[0].double()).view(1, 1, _H).float(), k, v

    SHAPES = {"gamma_in": (_H,), "w_qg": (1, _H, _QG), "w_k": (1, _H, _KV),
              "w_v": (1, _H, _KV), "gamma_q": (_D,), "gamma_k": (_D,),
              "w_o": (1, _OI, _H)}

    def make_weights(dtype):
        w = {}
        for n, sh in SHAPES.items():
            t = torch.randn(sh, device=DEV, dtype=torch.float32)
            w[n] = t if n.startswith("gamma") else (t * 0.02)
            if dtype is torch.bfloat16 and not n.startswith("gamma"):
                w[n] = w[n].bfloat16()
        return w

    cos_cache = torch.randn(_ROWS, _ROT, device=DEV)
    sin_cache = torch.randn(_ROWS, _ROT, device=DEV)
    scale = torch.full((1, 1, 1, 1), _D ** -0.5, device=DEV)

    print("=" * 74)
    print("(a) kernel vs plain torch")
    print("=" * 74)

    # --- partial_rope / partial_rope_kv (pure f32: < 1e-5) ---------------
    for name, fn, heads in (("partial_rope", partial_rope, _HQ),
                            ("partial_rope_kv", partial_rope_kv, _HKV)):
        x = torch.randn(1, 1, heads, _D, device=DEV)
        pid = torch.tensor([1234], device=DEV, dtype=torch.int32)
        e = relerr(fn(x, cos_cache, sin_cache, pid), t_rope(x, cos_cache, sin_cache, pid))
        print(f"  {name:22s} relerr {e:.3e}   {'PASS' if e < 1e-5 else 'FAIL'} (< 1e-5)")

    # --- full_attention, bf16 weights (< 2e-3) ---------------------------
    CS = (0, 1, 63, 64, 512, 2047)
    for dtype, tol in ((torch.bfloat16, 2e-3), (torch.float32, 2e-3)):
        w = make_weights(dtype)
        tag = "bf16" if dtype is torch.bfloat16 else "f32 "
        for C in CS:
            hidden = torch.randn(1, 1, _H, device=DEV)
            kc = torch.randn(1, C, _HKV, _D, device=DEV)
            vc = torch.randn(1, C, _HKV, _D, device=DEV)
            pid = torch.tensor([C], device=DEV, dtype=torch.int32)
            got = full_attention(hidden, w["gamma_in"], w["w_qg"], w["w_k"], w["w_v"],
                                 w["gamma_q"], w["gamma_k"], cos_cache, sin_cache,
                                 pid, kc, vc, scale, w["w_o"])
            ref = t_attn(hidden, w["gamma_in"], w["w_qg"], w["w_k"], w["w_v"],
                         w["gamma_q"], w["gamma_k"], cos_cache, sin_cache,
                         pid, kc, vc, scale, w["w_o"])
            es = [relerr(got[i], ref[i]) for i in range(3)]
            ok = max(es) < tol
            print(f"  full_attention {tag} C={C:5d}  out {es[0]:.3e}  k_rope {es[1]:.3e}"
                  f"  v {es[2]:.3e}   {'PASS' if ok else 'FAIL'} (< {tol:g})")

    # --- full_attention_fixed vs full_attention -------------------------
    print("-" * 74)
    CAP = 1024
    w = make_weights(torch.bfloat16)
    kcf = torch.full((1, CAP, _HKV, _D), float("nan"), device=DEV)
    vcf = torch.full((1, CAP, _HKV, _D), float("nan"), device=DEV)
    pos = torch.zeros(1, device=DEV, dtype=torch.int32)
    outf = torch.empty(1, 1, _H, device=DEV)
    hid = {p: torch.randn(1, 1, _H, device=DEV) for p in range(513)}
    CHECK = (0, 1, 63, 64, 512)
    for p in range(513):
        pos.fill_(p)
        full_attention_fixed(hid[p], w["gamma_in"], w["w_qg"], w["w_k"], w["w_v"],
                             w["gamma_q"], w["gamma_k"], cos_cache, sin_cache,
                             kcf, vcf, pos, scale, w["w_o"], outf)
        if p in CHECK:
            pid = torch.tensor([p], device=DEV, dtype=torch.int32)
            ref = full_attention(hid[p], w["gamma_in"], w["w_qg"], w["w_k"], w["w_v"],
                                 w["gamma_q"], w["gamma_k"], cos_cache, sin_cache,
                                 pid, kcf[:, :p].contiguous(), vcf[:, :p].contiguous(),
                                 scale, w["w_o"])
            e = relerr(outf, ref[0])
            nan = bool(torch.isnan(outf).any())
            print(f"  full_attention_fixed pos={p:4d}  vs full_attention {e:.3e}"
                  f"  nan={nan}   {'PASS' if (e < 2e-3 and not nan) else 'FAIL'}")
        if p in CHECK:
            # the slot it just wrote must be the k_rope/v full_attention returns
            assert not torch.isnan(kcf[0, p]).any(), "cache slot not written"

    # --- CUDA-graph capture with pos as a buffer -------------------------
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    hg = torch.randn(1, 1, _H, device=DEV)
    with torch.cuda.stream(s):
        for _ in range(3):
            full_attention_fixed(hg, w["gamma_in"], w["w_qg"], w["w_k"], w["w_v"],
                                 w["gamma_q"], w["gamma_k"], cos_cache, sin_cache,
                                 kcf, vcf, pos, scale, w["w_o"], outf)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        full_attention_fixed(hg, w["gamma_in"], w["w_qg"], w["w_k"], w["w_v"],
                             w["gamma_q"], w["gamma_k"], cos_cache, sin_cache,
                             kcf, vcf, pos, scale, w["w_o"], outf)
    worst = 0.0
    for p in (7, 100, 511):
        pos.fill_(p)
        g.replay()
        torch.cuda.synchronize()
        got = outf.clone()
        pid = torch.tensor([p], device=DEV, dtype=torch.int32)
        ref = full_attention(hg, w["gamma_in"], w["w_qg"], w["w_k"], w["w_v"],
                             w["gamma_q"], w["gamma_k"], cos_cache, sin_cache,
                             pid, kcf[:, :p].contiguous(), vcf[:, :p].contiguous(),
                             scale, w["w_o"])
        worst = max(worst, relerr(got, ref[0]))
    print(f"  graph replay tracks pos (7,100,511): worst relerr {worst:.3e}"
          f"   {'PASS' if worst < 2e-3 else 'FAIL'}")

    # ---- (b) against the authored reference -----------------------------
    print("=" * 74)
    print("(b) kernel vs the authored reference (f32 weights, DictResource)")
    print("=" * 74)
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import model  # noqa: E402
    from tilefoundry.runtime import DictResource  # noqa: E402

    mod = model.Qwen3_5FullAttention
    w32 = make_weights(torch.float32)
    loaded = mod.load(DictResource({n: w32[n] for n in mod.weights}))
    print("  the authored Module runs through the HIR interpreter, so only small")
    print("  C is compared here; large C is covered by the torch twin above.")
    for C in (0, 1, 63, 64):
        hidden = torch.randn(1, 1, _H, device=DEV)
        kc = torch.randn(1, C, _HKV, _D, device=DEV)
        vc = torch.randn(1, C, _HKV, _D, device=DEV)
        pid = torch.tensor([C], device=DEV, dtype=torch.int32)
        ref = loaded.full_attention(hidden, cos_cache, sin_cache, pid, kc, vc, scale)
        got = full_attention(hidden, w32["gamma_in"], w32["w_qg"], w32["w_k"],
                             w32["w_v"], w32["gamma_q"], w32["gamma_k"],
                             cos_cache, sin_cache, pid, kc, vc, scale, w32["w_o"])
        es = [relerr(got[i], ref[i]) for i in range(3)]
        print(f"  authored C={C:5d}  out {es[0]:.3e}  k_rope {es[1]:.3e}  v {es[2]:.3e}"
              f"   {'PASS' if max(es) < 2e-3 else 'FAIL'}")
    for name in ("partial_rope", "partial_rope_kv"):
        heads = _HQ if name == "partial_rope" else _HKV
        x = torch.randn(1, 1, heads, _D, device=DEV)
        pid = torch.tensor([77], device=DEV, dtype=torch.int32)
        ref = getattr(mod, name)(x, cos_cache, sin_cache, pid)
        got = globals()[name](x, cos_cache, sin_cache, pid)
        e = relerr(got, ref)
        print(f"  authored {name:16s} relerr {e:.3e}   {'PASS' if e < 1e-5 else 'FAIL'}")

    # ---- (c) timings, inside a CUDA graph -------------------------------
    print("=" * 74)
    print("(c) wall time per call, inside a CUDA graph (bf16 weights)")
    print("=" * 74)
    x16 = torch.randn(1, 1, _HQ, _D, device=DEV)
    x2 = torch.randn(1, 1, _HKV, _D, device=DEV)
    pid0 = torch.tensor([9], device=DEV, dtype=torch.int32)
    print(f"  partial_rope            {graph_bench(lambda: partial_rope(x16, cos_cache, sin_cache, pid0)):7.2f} us")
    print(f"  partial_rope_kv         {graph_bench(lambda: partial_rope_kv(x2, cos_cache, sin_cache, pid0)):7.2f} us")

    NC = 8  # rotating weight sets: 8 x 52 MB defeats the 60 MB L2
    wc = [make_weights(torch.bfloat16) for _ in range(NC)]
    hidden = torch.randn(1, 1, _H, device=DEV)
    print("  full_attention        warm     cold   (warm = one weight set, which"
          " fits in L2)")
    for C in (0, 64, 512, 2047):
        kc = torch.randn(1, C, _HKV, _D, device=DEV)
        vc = torch.randn(1, C, _HKV, _D, device=DEV)
        pid = torch.tensor([C], device=DEV, dtype=torch.int32)

        def one(i=0, kc=kc, vc=vc, pid=pid):
            u = wc[i % NC]
            return full_attention(hidden, u["gamma_in"], u["w_qg"], u["w_k"], u["w_v"],
                                  u["gamma_q"], u["gamma_k"], cos_cache, sin_cache,
                                  pid, kc, vc, scale, u["w_o"])

        warm = graph_bench(lambda: one(0))
        ctr = [0]

        def rot():
            ctr[0] += 1
            return one(ctr[0])

        cold = graph_bench(rot)
        _, ns, sl = _attn_kernel(
            C, C, _CFG_PROJ["bfloat16"][4], _proj_kernel("bfloat16")[1], False)
        print(f"    C={C:5d}  {warm:7.2f} {cold:8.2f} us   (attn grid {_HKV * ns:3d},"
              f" {sl} lse slots)")
    print("  per launch, cold (37.7 MB / attention / 16.8 MB; floor at 3.2 TB/s"
          " is 11.8 / ~3 / 5.2)")
    kern1, kskv = _proj_kernel("bfloat16")
    ksq = _CFG_PROJ["bfloat16"][4]
    hf_ = hidden.view(_H)
    qg_ = torch.empty(ksq, _QG, device=DEV)
    kr_ = torch.empty(kskv, _KV, device=DEV)
    vr_ = torch.empty(kskv, _KV, device=DEV)
    out_ = torch.zeros(_H, device=DEV)
    ct = [0]

    def k1call():
        ct[0] += 1
        u = wc[ct[0] % NC]
        kern1(hf_, u["gamma_in"], u["w_qg"].view(_H, _QG), u["w_k"].view(_H, _KV),
              u["w_v"].view(_H, _KV), qg_, kr_, vr_)

    print(f"    1 proj                {graph_bench(k1call):7.2f} us")
    for C in (0, 64, 512, 2047):
        k2, ns, sl = _attn_kernel(C, C, ksq, kskv, False)
        kc = torch.randn(max(C, 1), _HKV, _D, device=DEV)
        vc = torch.randn(max(C, 1), _HKV, _D, device=DEV)
        pid = torch.tensor([C], device=DEV, dtype=torch.int32)
        op = torch.empty(sl, _HQ, _D, device=DEV)
        mp = torch.empty(sl, _HQ, device=DEV)
        lp = torch.empty(sl, _HQ, device=DEV)
        kro = torch.empty(_HKV, _D, device=DEV)
        vo = torch.empty(_HKV, _D, device=DEV)
        t2 = graph_bench(lambda: k2(qg_, kr_, vr_, wc[0]["gamma_q"], wc[0]["gamma_k"],
                                    cos_cache, sin_cache, pid, kc, vc,
                                    scale.view(1), op, mp, lp, kro, vo, out_))
        k3 = _out_kernel("bfloat16", sl, ksq)
        c3 = [0]

        def k3call():
            c3[0] += 1
            k3(op, mp, lp, qg_, wc[c3[0] % NC]["w_o"].view(_OI, _H), out_)

        print(f"    2 attn C={C:5d} {t2:7.2f} us   3 out ({sl:2d} slots)"
              f" {graph_bench(k3call):7.2f} us")

    for cap in (1024,):
        pos.fill_(511)
        kcb = torch.zeros(1, cap, _HKV, _D, device=DEV)
        vcb = torch.zeros(1, cap, _HKV, _D, device=DEV)

        def onef(i=0):
            u = wc[i % NC]
            full_attention_fixed(hidden, u["gamma_in"], u["w_qg"], u["w_k"], u["w_v"],
                                 u["gamma_q"], u["gamma_k"], cos_cache, sin_cache,
                                 kcb, vcb, pos, scale, u["w_o"], outf)

        ctr2 = [0]

        def rotf():
            ctr2[0] += 1
            onef(ctr2[0])

        print(f"  full_attention_fixed CAP={cap} pos=511  "
              f"{graph_bench(lambda: onef(0)):7.2f} {graph_bench(rotf):8.2f} us")
        for p in (0, 1, 63, 64, 512):
            pos.fill_(p)
            print(f"    pos={p:5d}  {graph_bench(lambda: onef(0)):7.2f} us")
