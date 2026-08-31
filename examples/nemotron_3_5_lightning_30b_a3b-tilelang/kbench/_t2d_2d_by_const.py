
import tilelang, tilelang.language as T
N, THREADS = 4096, 128


@tilelang.jit
def build():
    @T.prim_func
    def main(ptrs: T.Tensor((4,), "int64"), stride: T.Tensor((1,), "int32"),
             out: T.Tensor((8,), "float32")):
        with T.Kernel(8, 4, threads=THREADS) as (bx, by):
            smem = T.alloc_shared((N,), "bfloat16")
            bar = T.alloc_barrier([1])
            tid = T.get_thread_binding()
            src = T.make_tensor_from_addr(ptrs[0], (1 << 20,), "bfloat16")
            st = T.Cast("int32", stride[0])
            for _b in T.serial(2):
                T.tma_copy(src[T.max(T.min((by + 4 * _b) * N, (1 << 20) - N), 0):T.max(T.min((by + 4 * _b) * N, (1 << 20) - N), 0) + N], smem[0:N], barrier=bar[0])
                if T.shuffle_elect(THREADS):
                    T.barrier_arrive(bar[0])
                T.mbarrier_wait_parity(bar[0], _b % 2)
                if tid < 8:
                    out[tid] = T.Cast("float32", smem[tid])
    return main
