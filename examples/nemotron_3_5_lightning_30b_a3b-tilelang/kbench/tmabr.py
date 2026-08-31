"""Does a 1-D bulk TMA still take the bulk path inside a blockIdx branch?"""
import sys, torch, tilelang
import tilelang.language as T
N, CTAS, THREADS = 4096, 8, 128
GUARD = len(sys.argv) > 1 and sys.argv[1] == "guard"


@tilelang.jit
def build():
    @T.prim_func
    def main(ptrs: T.Tensor((4,), "int64"), out: T.Tensor((8,), "float32")):
        with T.Kernel(CTAS, threads=THREADS) as bx:
            smem = T.alloc_shared((N,), "bfloat16")
            bar = T.alloc_barrier([1])
            tid = T.get_thread_binding()
            src = T.make_tensor_from_addr(ptrs[0], (1 << 20,), "bfloat16")
            if bx < 4:
                T.tma_copy(src[T.max(T.min(bx * N, (1 << 20) - N), 0):
                               T.max(T.min(bx * N, (1 << 20) - N), 0) + N],
                           smem[0:N], barrier=bar[0])
                if T.shuffle_elect(THREADS):
                    T.barrier_arrive(bar[0])
                T.mbarrier_wait_parity(bar[0], 0)
                if tid < 8:
                    out[tid] = T.Cast("float32", smem[tid])
    return main


if __name__ == "__main__":
    buf = torch.arange(1 << 20, device="cuda", dtype=torch.bfloat16)
    ptrs = torch.tensor([buf.data_ptr()] * 4, device="cuda", dtype=torch.int64)
    out = torch.zeros(8, device="cuda", dtype=torch.float32)
    build()(ptrs, out)
    torch.cuda.synchronize()
    print("ok:", out[:4].tolist())
