"""Can a kernel reach a buffer through an address in a table, and TMA out of it?"""
import torch
import tilelang
import tilelang.language as T

N = 4096
CTAS = 4
THREADS = 128


@tilelang.jit(out_idx=[-1])
def build():
    @T.prim_func
    def main(ptrs: T.Tensor((4,), "int64"), out: T.Tensor((4, 8), "float32")):
        with T.Kernel(CTAS, threads=THREADS) as bx:
            smem = T.alloc_shared((N,), "bfloat16")
            bar = T.alloc_barrier([1])
            acc = T.alloc_local((1,), "float32")
            tid = T.get_thread_binding()
            src = T.make_tensor_from_addr(ptrs[bx], (N,), "bfloat16")
            T.tma_copy(src[0:N], smem[0:N], barrier=bar[0])
            if T.shuffle_elect(THREADS):
                T.barrier_arrive(bar[0])
            T.mbarrier_wait_parity(bar[0], 0)
            acc[0] = 0.0
            for r in T.serial(N // THREADS):
                acc[0] += T.Cast("float32", smem[r * THREADS + tid])
            if tid < 8:
                out[bx, tid] = acc[0] + T.Cast("float32", smem[tid])
    return main


if __name__ == "__main__":
    fn = build()
    bufs = [torch.full((N,), float(i + 1), device="cuda", dtype=torch.bfloat16) for i in range(4)]
    ptrs = torch.tensor([b.data_ptr() for b in bufs], device="cuda", dtype=torch.int64)
    out = fn(ptrs)
    torch.cuda.synchronize()
    for i in range(4):
        want = (i + 1) * (N / THREADS) + (i + 1)
        print(f"buf {i}: got {out[i, 0].item():.1f} want {want:.1f}")
