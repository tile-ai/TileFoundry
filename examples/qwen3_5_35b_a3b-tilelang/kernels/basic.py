"""The primitives every boundary shares: GEMV, RMSNorm, gather, elementwise.

The GEMV is the whole ballgame. A decode step is 2.95 G weights read once, so
every matmul in `model.py` is a matrix-times-**vector** and the kernel is
memory-bound by definition: at H200's ~4.8 TB/s of HBM3e, one token's weights
are ~1.3 ms of pure traffic and the arithmetic is ~6 GFLOP, which is 6 us. There
is nothing to do but read the weights at line rate.

Why the GEMV is written as a GEMM with 15 dead rows
---------------------------------------------------
`T.gemm` is the only path in tilelang that gets the Hopper pipeline -- async
`T.copy` into multi-buffered shared memory, `wgmma`, swizzled layouts. A
hand-written "one output per thread, walk K" GEMV was measured at 2.6-3.7 TB/s
against `T.gemm`'s 4.4-4.8 TB/s on the same shapes (see WORKLOG 19:05), because
a scalar load per thread per k cannot keep enough requests in flight.

So the vector is placed in row 0 of a 16-row tile and rows 1..15 are left
**uninitialised**. This looks wrong and is not: an MMA computes each output row
from its own input row, so garbage in `xs[1:]` lands only in `acc[1:]`, which
nothing reads. Clearing them costs 15/16 of the shared-memory stores in the
prologue -- measured at 6.3 us -> 2.6 us on K=2048,N=512 -- and buys nothing.
The MMA itself does 16x the necessary flops, which is free at 6 us of arithmetic
against 1300 us of traffic.

The one thing that is *not* free is the epilogue: `acc` is a fragment whose
layout spreads both axes over threads, so `for j in T.Parallel(BN): Y[..] =
acc[0, j]` is rejected ("Loop layout is not injective"). Row 0 has to be reached
through shared memory.
"""
from __future__ import annotations

import functools

import tilelang
import tilelang.language as T
import torch

#: Rows of the MMA tile. 16 is the smallest the SM90 path accepts.
BM = 16

#: How much shared memory one SM may be given, minus a margin for the pipeline's
#: own bookkeeping. Used to pick a config that will actually compile rather than
#: to discover at runtime that it will not.
_SMEM_BUDGET = 220 * 1024


def _config(K: int, N: int) -> tuple[int, int, int, int]:
    """(BN, BK, threads, stages) for one GEMV shape.

    Chosen by the two things that were measured to matter: enough blocks to fill
    132 SMs (so `BN` shrinks as `N` shrinks), and a shared-memory footprint that
    fits (so `stages` shrinks as `K` grows, since the staged vector is `BM * K`).
    """
    threads = 128
    BK = 128
    # The vector tile is resident for the whole loop; the weight tile is staged.
    vector_bytes = BM * K * 2
    for BN in (128, 64):
        if N % BN:
            continue
        for stages in (4, 3, 2):
            if vector_bytes + stages * BK * BN * 2 + BM * BN * 4 <= _SMEM_BUDGET:
                return BN, BK, threads, stages
    # Nothing fitted with the vector resident; fall back to the narrowest tile.
    return (64 if N % 128 else 128), 64, threads, 2


@functools.lru_cache(maxsize=None)
def gemv(K: int, N: int, bias: bool = False, act: str = "none"):
    """`y[N] = x[K] @ W[K, N]`, x in f32, W in bf16, y in f32.

    *act* fuses the pointwise tail a caller would otherwise launch for:
      - `"none"`   y
      - `"silu"`   silu(y)
      - `"sigmoid"` sigmoid(y)
    """
    BN, BK, threads, stages = _config(K, N)
    KO = (K + BK - 1) // BK

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((K,), "float32"),
            W: T.Tensor((K, N), "bfloat16"),
            Y: T.Tensor((N,), "float32"),
        ):
            with T.Kernel(T.ceildiv(N, BN), threads=threads) as bn:
                xs = T.alloc_shared((BM, KO * BK), "bfloat16")
                ws = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                out = T.alloc_shared((BM, BN), "float32")
                # Row 0 only: see the module docstring. The tail past K is zeroed
                # so a K that is not a multiple of BK contributes nothing.
                for j in T.Parallel(KO * BK):
                    xs[0, j] = T.cast(X[j], "bfloat16") if j < K else T.cast(0.0, "bfloat16")
                T.clear(acc)
                T.sync_threads()
                for ko in T.Pipelined(KO, num_stages=stages):
                    T.copy(W[ko * BK:(ko + 1) * BK, bn * BN:(bn + 1) * BN], ws)
                    T.gemm(xs[:, ko * BK:(ko + 1) * BK], ws, acc)
                T.copy(acc, out)
                for j in T.Parallel(BN):
                    v = out[0, j]
                    if act == "silu":
                        Y[bn * BN + j] = v / (1.0 + T.exp(-v))
                    elif act == "sigmoid":
                        Y[bn * BN + j] = 1.0 / (1.0 + T.exp(-v))
                    else:
                        Y[bn * BN + j] = v

        return main

    return build()


@functools.lru_cache(maxsize=None)
def rms_norm(H: int, eps: float = 1e-6):
    """`y = x * rsqrt(mean(x^2) + eps) * gamma`, one block, f32 throughout.

    `tf.rms_norm` is `x * weight` flat -- no `1 +`. The published gammas carry
    the `1 +` already, folded in by `model.py`'s converters.
    """
    threads = 256 if H >= 256 else 128

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((H,), "float32"),
            G: T.Tensor((H,), "float32"),
            Y: T.Tensor((H,), "float32"),
        ):
            with T.Kernel(1, threads=threads) as _:
                # `T.reduce_sum` over a fragment is the only whole-block
                # reduction tilelang exposes; a hand-written tree needs a
                # `while` over halving widths, and the eager builder rejects a
                # `while` whose condition folds to a constant at build time.
                sq = T.alloc_fragment((H,), "float32")
                total = T.alloc_fragment((1,), "float32")
                scale = T.alloc_shared((1,), "float32")
                for i in T.Parallel(H):
                    sq[i] = X[i] * X[i]
                T.reduce_sum(sq, total, dim=0)
                if T.get_thread_binding() == 0:
                    scale[0] = T.rsqrt(total[0] / T.cast(H, "float32") + eps)  # eps baked
                T.sync_threads()
                for i in T.Parallel(H):
                    Y[i] = X[i] * scale[0] * G[i]

        return main

    return build()


@functools.lru_cache(maxsize=None)
def add(N: int):
    """`c = a + b`, f32. The layer's residual."""
    threads = 256
    PER = (N + threads - 1) // threads

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            A: T.Tensor((N,), "float32"),
            B: T.Tensor((N,), "float32"),
            C: T.Tensor((N,), "float32"),
        ):
            with T.Kernel(T.ceildiv(N, threads * PER), threads=threads) as b:
                tid = T.get_thread_binding()
                for p in T.serial(PER):
                    idx = b * threads * PER + p * threads + tid
                    if idx < N:
                        C[idx] = A[idx] + B[idx]

        return main

    return build()


@functools.lru_cache(maxsize=None)
def embed_row(V: int, H: int):
    """One row of the embedding table, by a token id held on the device.

    The id is a device tensor rather than a Python int so a decode step stays
    inside one CUDA graph: the token the previous step sampled never has to come
    back to the host.
    """
    threads = 256
    PER = (H + threads - 1) // threads

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Table: T.Tensor((V, H), "bfloat16"),
            Ids: T.Tensor((1,), "int64"),
            Y: T.Tensor((H,), "float32"),
        ):
            with T.Kernel(1, threads=threads) as _:
                tid = T.get_thread_binding()
                for p in T.serial(PER):
                    idx = p * threads + tid
                    # The `< V` guard is a bounds check and also the reason `V`
                    # resolves at all: tilelang evaluates the (stringified)
                    # annotations against the kernel function's *closure cells*,
                    # and Python only creates a cell for a name the nested
                    # function's code references. A dimension that appears only
                    # in an annotation raises `NameError`. tilelang's own
                    # `get_func_nonlocals` comment calls this out as known.
                    if idx < H and Ids[0] < V:
                        Y[idx] = T.cast(Table[Ids[0], idx], "float32")

        return main

    return build()


def torch_dtype(name: str) -> torch.dtype:
    return {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[name]


__all__ = ["BM", "add", "embed_row", "gemv", "rms_norm", "torch_dtype"]
