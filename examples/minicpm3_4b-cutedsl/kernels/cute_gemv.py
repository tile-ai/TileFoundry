"""The GEMV family: every projection on the decode path, in CuTeDSL.

At one token per step every projection is a matrix-vector product, so all of
these are memory-bound on the weight and the only question is how fast the
weight can be read. The shape is **one warp per output element**: the weight is
stored `(out, in)`, so that element's whole `in`-length row lies contiguous
under the warp, read 128 bits per lane at a time, and the dot product closes
with a shuffle reduction. Nothing is tiled, staged or double-buffered, because
there is no reuse to exploit -- each weight element is read once per token.

── Why split-K, when there is no reduction to speed up ──────────────────────

One warp per output element makes the *number of warps* equal to the number of
outputs, and this model's projections are narrow: `down_proj` has 2560 of them,
so 81920 threads, on a card that holds 270336. A third of the machine cannot
saturate HBM no matter how well each thread behaves, and the measurement says
so -- 3.1 TB/s where the same kernel shape over the 73448-row head reaches 4.2.

So a row is split across `KSPLIT` warps of the same block, each taking a
contiguous stretch of `in`, and their partials are summed through shared memory
before the epilogue. Nothing about the arithmetic changes; the point is only
that there are now `KSPLIT` times as many warps with loads in flight. The split
is inside one block, so it costs a barrier rather than a second kernel and a
partials buffer.

`KSPLIT = 1` generates exactly the un-split kernel, and the head keeps it: at
18362 blocks it was never short of warps.

── Why the shared pieces are `@cute.jit` ────────────────────────────────────

`range_constexpr` and a data-dependent `if` are rewritten by the DSL's AST
preprocessor, and the preprocessor only sees the body of a *decorated* function.
A plain Python helper called from a kernel is traced, not preprocessed, and dies
with "range_constexpr should be preprocessed by preprocessor". Marking a
device-side helper `@cute.jit` puts it back inside the rewriter, which is why
the ones below carry the decorator and fully annotated parameters.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cuda.bindings import driver as _cuda

from ._common import BF16, F32, Compiled, cdiv

#: 8 bf16 = one 128-bit load per lane; 32 lanes = a 512-byte contiguous slab.
VEC = 8


@cute.jit
def _block_sum(
    value: cutlass.Float32, tidx: cutlass.Int32, warps: cutlass.Constexpr,
    scratch: cute.Tensor,
) -> cutlass.Float32:
    v = cute.arch.warp_reduction_sum(value)
    if tidx % 32 == 0:
        scratch[tidx // 32] = v
    cute.arch.barrier()
    total = F32(0.0)
    for i in cutlass.range_constexpr(warps):
        total += scratch[i]
    return total


@cute.jit
def _load_and_normalise(
    x: cute.Tensor, gamma: cute.Tensor, smem: cute.Tensor, scratch: cute.Tensor,
    tidx: cutlass.Int32, k: cutlass.Constexpr, eps: cutlass.Constexpr,
    nt: cutlass.Constexpr,
) -> None:
    """Put `rms_norm(x, gamma)` into *smem* as bf16, in the order HF does it.

    `MiniCPMRMSNorm` lands the normalised activation in bf16 *before* the
    learned scale multiplies it, so the two roundings are kept apart here too --
    on bf16 they differ in the last bit, and the reference keeps them apart.
    """
    nch = k // VEC
    xv = cute.tiled_divide(x, (VEC,))
    gv = cute.tiled_divide(gamma, (VEC,))
    sv = cute.tiled_divide(smem, (VEC,))
    frag = cute.make_fragment(VEC, BF16)
    fg = cute.make_fragment(VEC, BF16)

    acc = F32(0.0)
    for i in cutlass.range_constexpr(cdiv(nch, nt)):
        j = i * nt + tidx
        if j < nch:
            cute.autovec_copy(xv[None, j], frag)
            for e in cutlass.range_constexpr(VEC):
                f = frag[e].to(F32)
                acc += f * f
    total = _block_sum(acc, tidx, nt // 32, scratch)
    inv = F32(cmath.rsqrt(total / F32(float(k)) + F32(eps)))

    for i in cutlass.range_constexpr(cdiv(nch, nt)):
        j = i * nt + tidx
        if j < nch:
            cute.autovec_copy(xv[None, j], frag)
            cute.autovec_copy(gv[None, j], fg)
            for e in cutlass.range_constexpr(VEC):
                landed = (frag[e].to(F32) * inv).to(BF16)
                frag[e] = (landed.to(F32) * fg[e].to(F32)).to(BF16)
            cute.autovec_copy(frag, sv[None, j])
    cute.arch.barrier()


@cute.jit
def _stage(
    x: cute.Tensor, smem: cute.Tensor, tidx: cutlass.Int32,
    k: cutlass.Constexpr, divisor: cutlass.Constexpr, nt: cutlass.Constexpr,
) -> None:
    """Copy *x* into *smem*, divided by *divisor* -- the no-norm prologue."""
    nch = k // VEC
    xv = cute.tiled_divide(x, (VEC,))
    sv = cute.tiled_divide(smem, (VEC,))
    frag = cute.make_fragment(VEC, BF16)
    for i in cutlass.range_constexpr(cdiv(nch, nt)):
        j = i * nt + tidx
        if j < nch:
            cute.autovec_copy(xv[None, j], frag)
            if cutlass.const_expr(divisor != 1.0):
                for e in cutlass.range_constexpr(VEC):
                    frag[e] = (frag[e].to(F32) / F32(divisor)).to(BF16)
            cute.autovec_copy(frag, sv[None, j])
    cute.arch.barrier()


#: Iterations of a row a warp will hold in registers to issue them early.
#: Measured, not guessed: at eight this is worth ~0.2 us on `q_a|kv_a`, and at
#: ten -- which is what `gate`/`up` together need -- the occupancy it costs
#: makes that kernel 0.6 us *slower*. So the two projections that fit stay
#: prefetched and the two that do not keep the plain loop.
MAX_PREFETCH = 8


def _iters(k: int, ksplit: int) -> int:
    return (k // ksplit) // (32 * VEC)


@cute.jit
def _issue(
    w: cute.Tensor, row: cutlass.Int32, lane: cutlass.Int32, kpart: cutlass.Int32,
    k: cutlass.Constexpr, ksplit: cutlass.Constexpr, iters: cutlass.Constexpr,
) -> cute.Tensor:
    """Start this warp's whole stretch of `w[row]` on its way to registers.

    Called *before* the norm, whose block-wide reduction ends in a barrier that
    no load may be hoisted across. The weight does not depend on the norm, so
    issuing it first means the two round trips overlap instead of queueing --
    which on the small projections is most of the kernel.
    """
    span = k // ksplit
    wv = cute.tiled_divide(w[row, None], (VEC,))
    fw = cute.make_fragment((VEC, iters), BF16)
    base = kpart * (span // VEC)
    for i in cutlass.range_constexpr(iters):
        cute.autovec_copy(wv[None, base + i * 32 + lane], fw[None, i])
    return fw


@cute.jit
def _consume(
    fw: cute.Tensor, smem: cute.Tensor, lane: cutlass.Int32, kpart: cutlass.Int32,
    k: cutlass.Constexpr, ksplit: cutlass.Constexpr, iters: cutlass.Constexpr,
) -> cutlass.Float32:
    """Finish the dot product from the weights `_issue` already fetched."""
    span = k // ksplit
    sv = cute.tiled_divide(smem, (VEC,))
    fx = cute.make_fragment(VEC, BF16)
    acc = F32(0.0)
    base = kpart * (span // VEC)
    for i in cutlass.range_constexpr(iters):
        cute.autovec_copy(sv[None, base + i * 32 + lane], fx)
        for e in cutlass.range_constexpr(VEC):
            acc += fw[e, i].to(F32) * fx[e].to(F32)
    return cute.arch.warp_reduction_sum(acc)


@cute.jit
def _part_dot(
    w: cute.Tensor, row: cutlass.Int32, smem: cute.Tensor, lane: cutlass.Int32,
    kpart: cutlass.Int32, k: cutlass.Constexpr, ksplit: cutlass.Constexpr,
) -> cutlass.Float32:
    """This warp's stretch of `dot(smem, w[row])`, accumulated in f32.

    `kpart` selects one of `ksplit` contiguous stretches of the row; with
    `ksplit == 1` it is the whole thing and the offset folds away.
    """
    span = k // ksplit
    sv = cute.tiled_divide(smem, (VEC,))
    wv = cute.tiled_divide(w[row, None], (VEC,))
    fw = cute.make_fragment(VEC, BF16)
    fx = cute.make_fragment(VEC, BF16)
    acc = F32(0.0)
    base = kpart * (span // VEC)
    for i in cutlass.range_constexpr(span // (32 * VEC)):
        j = base + i * 32 + lane
        cute.autovec_copy(wv[None, j], fw)
        cute.autovec_copy(sv[None, j], fx)
        for e in cutlass.range_constexpr(VEC):
            acc += fw[e].to(F32) * fx[e].to(F32)
    return cute.arch.warp_reduction_sum(acc)


@cute.jit
def _join(
    part: cutlass.Float32, scratch: cute.Tensor, warp: cutlass.Int32,
    lane: cutlass.Int32, slot: cutlass.Int32, ksplit: cutlass.Constexpr,
) -> cutlass.Float32:
    """Sum the `ksplit` warps that shared a row. Returns garbage off-slot."""
    if lane == 0:
        scratch[warp] = part
    cute.arch.barrier()
    total = F32(0.0)
    for i in cutlass.range_constexpr(ksplit):
        total += scratch[slot * ksplit + i]
    return total


def _plan(n: int, warps: int, ksplit: int, k: int, exact_rows: bool = True):
    """`(threads, rows per block, blocks)` for an `n`-row projection.

    Checked, not assumed. A `ksplit` that does not divide `k` into whole
    128-bit slabs would make `span // (32 * VEC)` truncate, and the kernel would
    quietly dot only part of each row -- a wrong answer with no symptom. Same
    for a `rows` that does not divide `n`: the last block would run off the end.
    """
    if warps % ksplit:
        raise ValueError(f"ksplit {ksplit} must divide warps {warps}")
    if k % (ksplit * 32 * VEC):
        raise ValueError(
            f"ksplit {ksplit} does not cut k={k} into whole {32 * VEC}-element "
            f"slabs; the dot product would silently drop the remainder"
        )
    rows = warps // ksplit
    if exact_rows and n % rows:
        raise ValueError(f"rows per block {rows} must divide n={n}")
    if warps * 32 > 1024:
        raise ValueError(f"{warps} warps is more than a block may have")
    return warps * 32, rows, cdiv(n, rows)


# ── rms_norm(x) @ w.T ────────────────────────────────────────────────────────

def _make_norm_gemv(k: int, n: int, eps: float, warps: int = 8, ksplit: int = 2):
    nt, rows, blocks = _plan(n, warps, ksplit, k)
    iters = _iters(k, ksplit)
    pre = iters <= MAX_PREFETCH

    @cute.kernel
    def kern(out: cute.Tensor, x: cute.Tensor, gamma: cute.Tensor, w: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane, warp = tidx % 32, tidx // 32
        smem = cute.make_tensor(cute.arch.alloc_smem(BF16, k, 16), cute.make_layout(k))
        scratch = cute.make_tensor(cute.arch.alloc_smem(F32, warps, 16),
                                   cute.make_layout(warps))
        row = bidx * rows + warp // ksplit
        kpart = warp % ksplit
        if cutlass.const_expr(pre):
            fw = _issue(w, row, lane, kpart, k, ksplit, iters)
            _load_and_normalise(x, gamma, smem, scratch, tidx, k, eps, nt)
            part = _consume(fw, smem, lane, kpart, k, ksplit, iters)
        else:
            _load_and_normalise(x, gamma, smem, scratch, tidx, k, eps, nt)
            part = _part_dot(w, row, smem, lane, kpart, k, ksplit)
        if cutlass.const_expr(ksplit == 1):
            if lane == 0:
                out[row] = part.to(BF16)
        else:
            cute.arch.barrier()
            value = _join(part, scratch, warp, lane, tidx, ksplit)
            if tidx < rows:
                out[bidx * rows + tidx] = value.to(BF16)

    @cute.jit
    def entry(out: cute.Tensor, x: cute.Tensor, gamma: cute.Tensor, w: cute.Tensor,
              stream: _cuda.CUstream):
        kern(out, x, gamma, w).launch(grid=[blocks, 1, 1], block=[nt, 1, 1],
                                      stream=stream)

    return entry


NORM_GEMV = Compiled(_make_norm_gemv)


# ── two of them, sharing nothing but a launch ───────────────────────────────

def _make_norm_gemv_pair(k1: int, n1: int, k2: int, n2: int, eps: float,
                         warps: int = 8, ksplit: int = 2):
    """`q_b` and `kv_b`: different inputs, different widths, one launch.

    They are siblings in the step -- neither reads the other -- so making them
    one grid costs a branch on `blockIdx` (uniform across a block, so free) and
    saves a kernel boundary, which over 62 layers is 60-odd microseconds a token.
    """
    nt, rows, b1 = _plan(n1, warps, ksplit, k1)
    _, _, b2 = _plan(n2, warps, ksplit, k2)
    kmax = max(k1, k2)

    @cute.jit
    def half(out: cute.Tensor, x: cute.Tensor, g: cute.Tensor, w: cute.Tensor,
             smem_ptr: cute.Pointer, scratch: cute.Tensor, tidx: cutlass.Int32,
             block: cutlass.Int32, k: cutlass.Constexpr) -> None:
        lane, warp = tidx % 32, tidx // 32
        smem = cute.make_tensor(smem_ptr, cute.make_layout(k))
        row = block * rows + warp // ksplit
        kpart = warp % ksplit
        its = (k // ksplit) // (32 * VEC)
        fw = _issue(w, row, lane, kpart, k, ksplit, its)
        _load_and_normalise(x, g, smem, scratch, tidx, k, eps, nt)
        part = _consume(fw, smem, lane, kpart, k, ksplit, its)
        if cutlass.const_expr(ksplit == 1):
            if lane == 0:
                out[row] = part.to(BF16)
        else:
            cute.arch.barrier()
            value = _join(part, scratch, warp, lane, tidx, ksplit)
            if tidx < rows:
                out[block * rows + tidx] = value.to(BF16)

    @cute.kernel
    def kern(out1: cute.Tensor, x1: cute.Tensor, g1: cute.Tensor, w1: cute.Tensor,
             out2: cute.Tensor, x2: cute.Tensor, g2: cute.Tensor, w2: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem_ptr = cute.arch.alloc_smem(BF16, kmax, 16)
        scratch = cute.make_tensor(cute.arch.alloc_smem(F32, warps, 16),
                                   cute.make_layout(warps))
        if bidx < b1:
            half(out1, x1, g1, w1, smem_ptr, scratch, tidx, bidx, k1)
        else:
            half(out2, x2, g2, w2, smem_ptr, scratch, tidx, bidx - b1, k2)

    @cute.jit
    def entry(out1: cute.Tensor, x1: cute.Tensor, g1: cute.Tensor, w1: cute.Tensor,
              out2: cute.Tensor, x2: cute.Tensor, g2: cute.Tensor, w2: cute.Tensor,
              stream: _cuda.CUstream):
        kern(out1, x1, g1, w1, out2, x2, g2, w2).launch(
            grid=[b1 + b2, 1, 1], block=[nt, 1, 1], stream=stream)

    return entry


NORM_GEMV_PAIR = Compiled(_make_norm_gemv_pair)


# ── rms_norm -> gate / up -> silu * mul, one launch ─────────────────────────

def _make_norm_gemv_swiglu(k: int, n: int, eps: float, warps: int = 8,
                           ksplit: int = 2):
    """`silu(n @ w_gate.T) * (n @ w_up.T)`.

    A warp owns output element *i* and reads **both** weight rows for it, so the
    product happens in registers: the two 6400-wide projections are never
    written to memory at all, which is 50 MB of round trip a two-kernel spelling
    would pay for.
    """
    nt, rows, blocks = _plan(n, warps, ksplit, k)
    iters = _iters(k, ksplit)
    pre = 2 * iters <= MAX_PREFETCH

    @cute.kernel
    def kern(out: cute.Tensor, x: cute.Tensor, gamma: cute.Tensor,
             w_gate: cute.Tensor, w_up: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane, warp = tidx % 32, tidx // 32
        smem = cute.make_tensor(cute.arch.alloc_smem(BF16, k, 16), cute.make_layout(k))
        sg = cute.make_tensor(cute.arch.alloc_smem(F32, warps, 16),
                              cute.make_layout(warps))
        su = cute.make_tensor(cute.arch.alloc_smem(F32, warps, 16),
                              cute.make_layout(warps))
        row = bidx * rows + warp // ksplit
        kpart = warp % ksplit
        if cutlass.const_expr(pre):
            fg = _issue(w_gate, row, lane, kpart, k, ksplit, iters)
            fu = _issue(w_up, row, lane, kpart, k, ksplit, iters)
            _load_and_normalise(x, gamma, smem, sg, tidx, k, eps, nt)
            pg = _consume(fg, smem, lane, kpart, k, ksplit, iters)
            pu = _consume(fu, smem, lane, kpart, k, ksplit, iters)
        else:
            _load_and_normalise(x, gamma, smem, sg, tidx, k, eps, nt)
            pg = _part_dot(w_gate, row, smem, lane, kpart, k, ksplit)
            pu = _part_dot(w_up, row, smem, lane, kpart, k, ksplit)
        if cutlass.const_expr(ksplit == 1):
            if lane == 0:
                _swiglu_store(out, row, pg, pu)
        else:
            cute.arch.barrier()
            if lane == 0:
                sg[warp] = pg
                su[warp] = pu
            cute.arch.barrier()
            if tidx < rows:
                g = F32(0.0)
                u = F32(0.0)
                for i in cutlass.range_constexpr(ksplit):
                    g += sg[tidx * ksplit + i]
                    u += su[tidx * ksplit + i]
                _swiglu_store(out, bidx * rows + tidx, g, u)

    @cute.jit
    def entry(out: cute.Tensor, x: cute.Tensor, gamma: cute.Tensor,
              w_gate: cute.Tensor, w_up: cute.Tensor, stream: _cuda.CUstream):
        kern(out, x, gamma, w_gate, w_up).launch(
            grid=[blocks, 1, 1], block=[nt, 1, 1], stream=stream)

    return entry


@cute.jit
def _swiglu_store(out: cute.Tensor, row: cutlass.Int32, g: cutlass.Float32,
                  u: cutlass.Float32) -> None:
    # Each projection lands in bf16 before the activation, exactly as the
    # reference's two separate `matmul`s do.
    gb = g.to(BF16).to(F32)
    ub = u.to(BF16).to(F32)
    act = gb / (F32(1.0) + F32(cmath.exp(-gb)))
    out[row] = (act.to(BF16).to(F32) * ub).to(BF16)


NORM_GEMV_SWIGLU = Compiled(_make_norm_gemv_swiglu)


# ── plain and residual GEMV ─────────────────────────────────────────────────

def _make_gemv(k: int, n: int, residual: bool, divisor: float = 1.0,
               warps: int = 8, ksplit: int = 2):
    """`x @ w.T`, optionally `+ residual * alpha` in the epilogue.

    The scaled residual is `scale_depth`'s, and it is nearly free here: the one
    lane that produced an output element is the one that needs the residual
    element, so it costs a single load and a single fma at the end of a row --
    against a separate elementwise kernel, which would be another round trip
    and, far worse, another boundary in a chain 372 kernels long.
    """
    nt, rows, blocks = _plan(n, warps, ksplit, k)
    iters = _iters(k, ksplit)
    pre = iters <= MAX_PREFETCH

    @cute.kernel
    def kern(out: cute.Tensor, x: cute.Tensor, w: cute.Tensor,
             res: cute.Tensor, alpha: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane, warp = tidx % 32, tidx // 32
        smem = cute.make_tensor(cute.arch.alloc_smem(BF16, k, 16), cute.make_layout(k))
        scratch = cute.make_tensor(cute.arch.alloc_smem(F32, warps, 16),
                                   cute.make_layout(warps))
        row = bidx * rows + warp // ksplit
        kpart = warp % ksplit
        if cutlass.const_expr(pre):
            fw = _issue(w, row, lane, kpart, k, ksplit, iters)
            _stage(x, smem, tidx, k, divisor, nt)
            part = _consume(fw, smem, lane, kpart, k, ksplit, iters)
        else:
            _stage(x, smem, tidx, k, divisor, nt)
            part = _part_dot(w, row, smem, lane, kpart, k, ksplit)
        if cutlass.const_expr(ksplit == 1):
            if lane == 0:
                _gemv_store(out, row, part, res, alpha, residual)
        else:
            if lane == 0:
                scratch[warp] = part
            cute.arch.barrier()
            if tidx < rows:
                total = F32(0.0)
                for i in cutlass.range_constexpr(ksplit):
                    total += scratch[tidx * ksplit + i]
                _gemv_store(out, bidx * rows + tidx, total, res, alpha, residual)

    @cute.jit
    def entry(out: cute.Tensor, x: cute.Tensor, w: cute.Tensor, res: cute.Tensor,
              alpha: cute.Tensor, stream: _cuda.CUstream):
        kern(out, x, w, res, alpha).launch(grid=[blocks, 1, 1], block=[nt, 1, 1],
                                           stream=stream)

    return entry


@cute.jit
def _gemv_store(out: cute.Tensor, row: cutlass.Int32, value: cutlass.Float32,
                res: cute.Tensor, alpha: cute.Tensor,
                residual: cutlass.Constexpr) -> None:
    if cutlass.const_expr(residual):
        scaled = (value.to(BF16).to(F32) * alpha[0].to(F32)).to(BF16)
        out[row] = (res[row].to(F32) + scaled.to(F32)).to(BF16)
    else:
        out[row] = value.to(BF16)


GEMV = Compiled(_make_gemv)


# ── the head ─────────────────────────────────────────────────────────────────

def _make_lm_head(k: int, n: int, divisor: float, warps: int = 4, ksplit: int = 1):
    """`(hidden / logits_scaling) @ w_head.T` over the whole vocabulary.

    73448 rows of 2560 is 359 MB -- by far the largest single read of a step,
    and the one kernel with no shortage of warps, so it keeps `ksplit = 1` and
    reaches the card's streaming bandwidth as it stands. The divide is a divide
    and not a multiply by 0.1, because `MiniCPM3ForCausalLM` divides and 1/10 is
    not a bf16 number.
    """
    nt, rows, blocks = _plan(n, warps, ksplit, k, exact_rows=False)

    @cute.kernel
    def kern(out: cute.Tensor, x: cute.Tensor, w: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cute.make_tensor(cute.arch.alloc_smem(BF16, k, 16), cute.make_layout(k))
        _stage(x, smem, tidx, k, divisor, nt)
        row = bidx * rows + tidx // 32
        if row < n:
            value = _part_dot(w, row, smem, tidx % 32, 0, k, 1)
            if tidx % 32 == 0:
                out[row] = value.to(BF16)

    @cute.jit
    def entry(out: cute.Tensor, x: cute.Tensor, w: cute.Tensor,
              stream: _cuda.CUstream):
        kern(out, x, w).launch(grid=[blocks, 1, 1], block=[nt, 1, 1], stream=stream)

    return entry


LM_HEAD = Compiled(_make_lm_head)

__all__ = ["GEMV", "LM_HEAD", "NORM_GEMV", "NORM_GEMV_PAIR", "NORM_GEMV_SWIGLU"]
