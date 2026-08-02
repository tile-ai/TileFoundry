"""The two kernels that bracket the stack: the scaled embedding row, and the
final RMSNorm.

Both touch a single 2560-wide vector, so both are one block. They exist as
kernels at all rather than as torch one-liners because a decode step is captured
as one CUDA graph, and a torch op in the middle of it is another node with
another set of allocations behind it.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cuda.bindings import driver as _cuda

from ._common import BF16, F32, Compiled, block_reduce_sum, cdiv

VEC = 8
WARPS = 8
NT = WARPS * 32


def _make_embed(hidden: int, scale: float):
    """`w_embed[token] * scale_emb` -- HF's `MiniCPM3ScaledWordEmbedding`."""
    nch = hidden // VEC

    @cute.kernel
    def kern(out: cute.Tensor, table: cute.Tensor, token: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        row = token[0]
        ov = cute.tiled_divide(out, (VEC,))
        frag = cute.make_fragment(VEC, BF16)
        for i in cutlass.range_constexpr(cdiv(nch, NT)):
            j = i * NT + tidx
            if j < nch:
                for e in cutlass.range_constexpr(VEC):
                    v = table[row, j * VEC + e].to(F32) * F32(scale)
                    frag[e] = v.to(BF16)
                cute.autovec_copy(frag, ov[None, j])

    @cute.jit
    def entry(out: cute.Tensor, table: cute.Tensor, token: cute.Tensor,
              stream: _cuda.CUstream):
        kern(out, table, token).launch(grid=[1, 1, 1], block=[NT, 1, 1], stream=stream)

    return entry


EMBED = Compiled(_make_embed)


def _make_rmsnorm(hidden: int, eps: float):
    """`MiniCPM3Model.norm`, the one RMSNorm with nothing fused onto it."""
    nch = hidden // VEC

    @cute.kernel
    def kern(out: cute.Tensor, x: cute.Tensor, gamma: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        scratch = cute.make_tensor(cute.arch.alloc_smem(F32, WARPS, 4),
                                   cute.make_layout(WARPS))
        xv = cute.tiled_divide(x, (VEC,))
        gv = cute.tiled_divide(gamma, (VEC,))
        ov = cute.tiled_divide(out, (VEC,))
        frag = cute.make_fragment(VEC, BF16)
        fg = cute.make_fragment(VEC, BF16)

        acc = F32(0.0)
        for i in cutlass.range_constexpr(cdiv(nch, NT)):
            j = i * NT + tidx
            if j < nch:
                cute.autovec_copy(xv[None, j], frag)
                for e in cutlass.range_constexpr(VEC):
                    f = frag[e].to(F32)
                    acc += f * f
        total = block_reduce_sum(acc, tidx, WARPS, scratch)
        inv = F32(cmath.rsqrt(total / F32(float(hidden)) + F32(eps)))

        for i in cutlass.range_constexpr(cdiv(nch, NT)):
            j = i * NT + tidx
            if j < nch:
                cute.autovec_copy(xv[None, j], frag)
                cute.autovec_copy(gv[None, j], fg)
                for e in cutlass.range_constexpr(VEC):
                    landed = (frag[e].to(F32) * inv).to(BF16)
                    frag[e] = (landed.to(F32) * fg[e].to(F32)).to(BF16)
                cute.autovec_copy(frag, ov[None, j])

    @cute.jit
    def entry(out: cute.Tensor, x: cute.Tensor, gamma: cute.Tensor,
              stream: _cuda.CUstream):
        kern(out, x, gamma).launch(grid=[1, 1, 1], block=[NT, 1, 1], stream=stream)

    return entry


RMSNORM = Compiled(_make_rmsnorm)

__all__ = ["EMBED", "RMSNORM"]
