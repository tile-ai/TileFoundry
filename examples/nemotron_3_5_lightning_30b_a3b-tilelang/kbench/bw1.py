"""How fast can a persistent CTA grid stream weights out of HBM?

Explicit TMA + mbarrier, a hand-rolled multi-stage prefetch, one CTA per SM.
"""
import sys, traceback, torch, tilelang, tilelang.language as T

CTAS = 132


def make(tile_rows, K, stages, threads, rows_total, grid_sync=0):
    per_cta = rows_total // CTAS
    ntiles = per_cta // tile_rows
    tile_elems = tile_rows * K

    @tilelang.jit(out_idx=[-1])
    def build():
        @T.prim_func
        def main(W: T.Tensor((rows_total, K), "bfloat16"),
                 acc: T.Tensor((CTAS,), "float32")):
            with T.Kernel(CTAS, threads=threads) as bx:
                smem = T.alloc_shared((stages, tile_rows, K), "bfloat16")
                bar = T.alloc_barrier([1] * stages)
                s = T.alloc_local((1,), "float32")
                tid = T.get_thread_binding()
                s[0] = 0.0
                base = bx * per_cta
                for j in T.serial(stages):
                    r0 = base + j * tile_rows
                    T.tma_copy(W[r0:r0 + tile_rows, 0:K], smem[j, :, :], barrier=bar[j])
                for i in T.serial(ntiles):
                    st = i % stages
                    T.mbarrier_wait_parity(bar[st], (i // stages) % 2)
                    for r in T.serial(tile_elems // threads):
                        idx = r * threads + tid
                        s[0] += T.Cast("float32", smem[st, idx // K, idx % K])
                    T.sync_threads()
                    if i + stages < ntiles:
                        r0 = base + (i + stages) * tile_rows
                        T.tma_copy(W[r0:r0 + tile_rows, 0:K], smem[st, :, :], barrier=bar[st])
                    if grid_sync == 1:
                        T.sync_grid()
                if tid == 0:
                    acc[bx] = s[0]
        return main
    return build()


def bench(fn, W, iters=20):
    out = fn(W)
    torch.cuda.synchronize()
    for _ in range(5):
        fn(W)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn(W)
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / iters
    gb = W.numel() * 2 / 1e9
    return ms, gb / (ms / 1e3) / 1e3, out


if __name__ == "__main__":
    K = 2688
    rows_total = CTAS * 8 * 256
    W = torch.randn(rows_total, K, device="cuda", dtype=torch.bfloat16)
    print(f"buffer {W.numel()*2/1e9:.2f} GB, rows {rows_total}")
    for tile_rows, stages, threads, gs in [(8, 4, 256, 0), (8, 6, 256, 0), (16, 4, 256, 0),
                                           (4, 8, 256, 0), (8, 4, 512, 0), (16, 3, 128, 0),
                                           (8, 8, 256, 0), (8, 4, 256, 1)]:
        try:
            fn = make(tile_rows, K, stages, threads, rows_total, gs)
            ms, tbs, _ = bench(fn, W)
            print(f"tile_rows={tile_rows:3d} stages={stages} threads={threads:4d} gsync={gs}  "
                  f"{ms:8.3f} ms  {tbs:6.3f} TB/s")
        except Exception as e:
            traceback.print_exc(); print("FAIL", tile_rows, stages, threads)
