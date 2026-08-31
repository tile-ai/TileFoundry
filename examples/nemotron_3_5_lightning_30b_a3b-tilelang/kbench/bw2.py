"""Persistent-grid HBM streaming, rank-1 shared buffers so the copy stays 1-D TMA."""
import sys, traceback, torch, tilelang, tilelang.language as T

CTAS = 132


def make(tile_elems, stages, threads, total, grid_sync=0, vec=8):
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
                    T.tma_copy(W[g:g + tile_elems], smem[j * tile_elems:(j + 1) * tile_elems],
                               barrier=bar[j])
                for i in T.serial(ntiles):
                    st = i % stages
                    T.mbarrier_wait_parity(bar[st], (i // stages) % 2)
                    o = st * tile_elems
                    for r in T.serial(tile_elems // (threads * vec)):
                        p = o + (r * threads + tid) * vec
                        for q in T.vectorized(vec):
                            s[0] += T.Cast("float32", smem[p + q])
                    T.sync_threads()
                    if i + stages < ntiles:
                        g = base + (i + stages) * tile_elems
                        T.tma_copy(W[g:g + tile_elems], smem[o:o + tile_elems], barrier=bar[st])
                    if grid_sync == 1:
                        T.sync_grid()
                if tid == 0:
                    acc[bx] = s[0]
        return main
    return build()


def bench(fn, W, iters=20):
    fn(W); torch.cuda.synchronize()
    for _ in range(5):
        fn(W)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn(W)
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / iters
    return ms, W.numel() * 2 / 1e9 / (ms / 1e3) / 1e3


if __name__ == "__main__":
    total = CTAS * 8192 * 256
    W = torch.randn(total, device="cuda", dtype=torch.bfloat16)
    print(f"buffer {W.numel()*2/1e9:.3f} GB")
    for tile_elems, stages, threads, gs in [
            (8192, 4, 256, 0), (8192, 6, 256, 0), (16384, 4, 256, 0),
            (4096, 8, 256, 0), (8192, 4, 512, 0), (16384, 6, 256, 0),
            (32768, 3, 256, 0), (8192, 8, 128, 0), (8192, 6, 256, 1)]:
        try:
            fn = make(tile_elems, stages, threads, total, gs)
            ms, tbs = bench(fn, W)
            print(f"tile={tile_elems:6d}elem stages={stages} thr={threads:4d} gsync={gs}  "
                  f"{ms:8.3f} ms  {tbs:6.3f} TB/s")
        except Exception as e:
            print("FAIL", tile_elems, stages, threads, gs, ":", str(e).splitlines()[-1][:200])
