"""What every CuTeDSL kernel here needs: a stream, a compile cache, block-wide
reductions.

**The stream is the reason this file exists.** A `@cute.jit` function launches
on stream 0 unless it is handed one, and stream 0 is not the stream a CUDA graph
captures. Every kernel in this package therefore takes a trailing
`stream: CUstream` parameter and every call site passes the *current* torch
stream -- which is the capture stream while a graph is being recorded, and the
ordinary one otherwise. Without that, capture records nothing and replays an
empty graph, silently.
"""
from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as _cuda
from cutlass.cute.runtime import from_dlpack

BF16 = cutlass.BFloat16
F32 = cutlass.Float32


def stream() -> "_cuda.CUstream":
    """The stream torch is currently on -- the capture stream during capture."""
    return _cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def t(x: torch.Tensor) -> cute.Tensor:
    """A torch tensor as a CuTeDSL one. `assumed_align=16` is what lets
    `autovec_copy` widen to 128-bit loads instead of element-sized ones."""
    return from_dlpack(x, assumed_align=16)


def t_dyn(x: torch.Tensor) -> cute.Tensor:
    """The same, with the extent left out of the compiled signature.

    For the KV cache: its length is the one thing about the decode path that
    changes, and baking it would mean a fresh compile per context length.
    """
    return from_dlpack(x, assumed_align=16).mark_layout_dynamic()


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class Compiled:
    """One `@cute.jit` entry, compiled on first call and kept.

    Compilation is per *shape*, so the key is every constexpr the factory takes;
    the tensors themselves are runtime arguments and may move between calls.
    """

    def __init__(self, factory, dynamic: tuple[int, ...] = ()) -> None:
        self._factory = factory
        self._dynamic = frozenset(dynamic)
        self._cache: dict = {}

    def _args(self, tensors):
        return [t_dyn(x) if i in self._dynamic else t(x) for i, x in enumerate(tensors)]

    def __call__(self, key, *tensors):
        entry = self._cache.get(key)
        if entry is None:
            entry = cute.compile(self._factory(*key), *self._args(tensors), stream())
            self._cache[key] = entry
        entry(*self._args(tensors), stream())


# ── device-side helpers, inlined into a kernel body ──────────────────────────

@cute.jit
def block_reduce_sum(
    value: cutlass.Float32, tidx: cutlass.Int32, warps: cutlass.Constexpr,
    scratch: cute.Tensor,
) -> cutlass.Float32:
    """Sum *value* across the whole block. *scratch* is `warps` f32 in smem.

    `@cute.jit` because the `if` below is on a dynamic value: a device helper
    only gets the AST rewriter -- and therefore `scf.if` instead of a Python
    `bool()` on a traced Boolean -- when it is decorated.
    """
    v = cute.arch.warp_reduction_sum(value)
    if tidx % 32 == 0:
        scratch[tidx // 32] = v
    cute.arch.barrier()
    total = F32(0.0)
    for i in cutlass.range_constexpr(warps):
        total += scratch[i]
    return total
