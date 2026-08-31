"""What one grid barrier costs, alone and while the memory pipe is kept busy."""
from __future__ import annotations

import argparse
import torch
import tilelang
import tilelang.language as T

CTAS = 132


def make_bare(n, threads):
    @tilelang.jit(out_idx=[-1])
    def build():
        @T.prim_func
        def main(acc: T.Tensor((CTAS,), "float32")):
            with T.Kernel(CTAS, threads=threads) as bx:
                for _ in T.serial(n):
                    T.sync_grid()
                acc[bx] = 1.0
        return main
    return build()


def make_streamed(n, tile_elems, stages, threads, total, tiles_per_sync):
    """One grid barrier every `tiles_per_sync` tiles, with the prefetch running past it."""
    per_cta = total // CTAS
    ntiles = per_cta // tile_elems

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
                    for r in T.serial(tile_elems // (threads * 8)):
                        p = o + (r * threads + tid) * 8
                        for q in T.vectorized(8):
                            s[0] += T.Cast("float32", smem[p + q])
                    T.sync_threads()
                    if i + stages < ntiles:
                        g = base + (i + stages) * tile_elems
                        T.tma_copy(W[g:g + tile_elems], smem[o:o + tile_elems], barrier=bar[st])
                        if T.shuffle_elect(threads):
                            T.barrier_arrive(bar[st])
                    if i % tiles_per_sync == tiles_per_sync - 1:
                        T.sync_grid()
                acc[bx] = s[0]
        return main
    return build()


def bench(fn, args, iters=20):
    fn(*args)
    torch.cuda.synchronize()
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bare")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--tile", type=int, default=8192)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--per-sync", type=int, default=8)
    ap.add_argument("--gb", type=float, default=4.0)
    a = ap.parse_args()
    if a.mode == "bare":
        for n in (0, a.n, 2 * a.n):
            fn = make_bare(n, a.threads)
            ms = bench(fn, (), iters=50)
            print(f"bare syncs={n:6d} threads={a.threads}  {ms:8.4f} ms"
                  f"  {ms * 1e6 / max(n, 1):7.3f} ns/sync", flush=True)
    else:
        per_cta_tiles = max(1, int(a.gb * 1e9 / 2 / CTAS / a.tile))
        total = CTAS * per_cta_tiles * a.tile
        W = torch.randn(total, device="cuda", dtype=torch.bfloat16)
        fn = make_streamed(a.n, a.tile, a.stages, a.threads, total, a.per_sync)
        ms = bench(fn, (W,))
        nsync = per_cta_tiles // a.per_sync
        tbs = total * 2 / 1e9 / (ms / 1e3) / 1e3
        print(f"streamed per_sync={a.per_sync} syncs={nsync} bytes={total*2/1e9:.2f}GB"
              f"  {ms:8.3f} ms  {tbs:6.3f} TB/s", flush=True)
