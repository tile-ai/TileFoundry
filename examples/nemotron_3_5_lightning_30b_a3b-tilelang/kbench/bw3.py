"""What a persistent cooperative grid can pull out of HBM, and what a grid barrier costs.

Explicit 1-D TMA (`cp.async.bulk`) into rank-1 shared buffers, an mbarrier per
stage, a hand-rolled N-deep prefetch. No `T.copy`, no `T.Pipelined`.
"""
from __future__ import annotations

import argparse
import torch
import tilelang
import tilelang.language as T

CTAS = 132


def make(tile_elems: int, stages: int, threads: int, total: int,
         grid_sync: int, vec: int = 8):
    per_cta = total // CTAS
    ntiles = per_cta // tile_elems
    inner = tile_elems // (threads * vec)

    @tilelang.jit(out_idx=[-1])
    def build():
        @T.prim_func
        def main(W: T.Tensor((total,), "bfloat16"), acc: T.Tensor((CTAS,), "float32")):
            with T.Kernel(CTAS, threads=threads) as bx:
                smem = T.alloc_shared((stages * tile_elems,), "bfloat16")
                bar = T.alloc_barrier([1] * stages)
                s = T.alloc_local((1,), "float32")
                tid = T.get_thread_binding()
                s[0] = 0.0
                base = bx * per_cta
                for j in T.serial(stages):
                    g = base + j * tile_elems
                    T.tma_copy(W[g:g + tile_elems],
                               smem[j * tile_elems:(j + 1) * tile_elems], barrier=bar[j])
                    if T.shuffle_elect(threads):
                        T.barrier_arrive(bar[j])
                for i in T.serial(ntiles):
                    st = i % stages
                    T.mbarrier_wait_parity(bar[st], (i // stages) % 2)
                    o = st * tile_elems
                    for r in T.serial(inner):
                        p = o + (r * threads + tid) * vec
                        for q in T.vectorized(vec):
                            s[0] += T.Cast("float32", smem[p + q])
                    T.sync_threads()
                    if i + stages < ntiles:
                        g = base + (i + stages) * tile_elems
                        T.tma_copy(W[g:g + tile_elems], smem[o:o + tile_elems], barrier=bar[st])
                        if T.shuffle_elect(threads):
                            T.barrier_arrive(bar[st])
                    if grid_sync > 0:
                        T.sync_grid()
                acc[bx] = s[0]
        return main
    return build()


def bench(fn, W, iters=20):
    fn(W)
    torch.cuda.synchronize()
    for _ in range(3):
        fn(W)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn(W)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / iters
    return ms, W.numel() * 2 / 1e9 / (ms / 1e3) / 1e3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=8192)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--gsync", type=int, default=0)
    ap.add_argument("--gb", type=float, default=4.0)
    a = ap.parse_args()
    per_cta_tiles = max(1, int(a.gb * 1e9 / 2 / CTAS / a.tile))
    total = CTAS * per_cta_tiles * a.tile
    W = torch.randn(total, device="cuda", dtype=torch.bfloat16)
    fn = make(a.tile, a.stages, a.threads, total, a.gsync)
    ms, tbs = bench(fn, W)
    nsync = per_cta_tiles if a.gsync else 0
    print(f"tile={a.tile} stages={a.stages} threads={a.threads} gsync={a.gsync} "
          f"bytes={total*2/1e9:.2f}GB syncs={nsync}  {ms:8.3f} ms  {tbs:6.3f} TB/s", flush=True)
