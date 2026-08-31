
import tilelang, tilelang.language as T
CTAS, THREADS = 132, 256
@tilelang.jit
def build():
    @T.prim_func
    def main(w: T.Tensor((637034496,), "bfloat16"), x: T.Tensor((2688,), "float32"),
             sel: T.Tensor((23,), "int32"), out: T.Tensor((10304,), "float32")):
        with T.Kernel(CTAS, threads=THREADS) as cta:
            sm = T.alloc_shared((43008,), "bfloat16")
            fs = T.alloc_shared((2688,), "float32")
            red = T.alloc_shared((THREADS,), "float32")
            pad_ = T.alloc_shared((1,), "float32")
            bar = T.alloc_barrier([1] * 2)
            acc = T.alloc_local((1,), "float32")
            tid = T.get_thread_binding()
            pad_[tid] = 0.0
            for _i in T.serial(11):
                if _i * THREADS + tid < 2688:
                    fs[_i * THREADS + tid] = x[_i * THREADS + tid]
            for _rep in T.serial(23):
                T.sync_threads()
                T.tma_copy(w[T.max(T.min(sel[_rep] * 27697152 + T.max(T.min(cta * 79 + (0) * 8, 10296), 0) * 2688, 637012992), 0):T.max(T.min(sel[_rep] * 27697152 + T.max(T.min(cta * 79 + (0) * 8, 10296), 0) * 2688, 637012992), 0) + 21504], sm[0:21504], barrier=bar[0])
                if T.shuffle_elect(THREADS):
                    T.barrier_arrive(bar[0])
                T.sync_threads()
                T.tma_copy(w[T.max(T.min(sel[_rep] * 27697152 + T.max(T.min(cta * 79 + (1) * 8, 10296), 0) * 2688, 637012992), 0):T.max(T.min(sel[_rep] * 27697152 + T.max(T.min(cta * 79 + (1) * 8, 10296), 0) * 2688, 637012992), 0) + 21504], sm[21504:43008], barrier=bar[1])
                if T.shuffle_elect(THREADS):
                    T.barrier_arrive(bar[1])
                T.sync_threads()
                for _t in T.serial(10):
                    T.mbarrier_wait_parity(bar[_t % 2], (_t // 2) % 2)
                    acc[0] = 0.0
                    for _i in T.serial(42):
                        acc[0] += fs[(tid % 32) * 2 + _i * 64 + 0] * T.Cast("float32", sm[(_t % 2) * 21504 + (tid // 32) * 2688 + (tid % 32) * 2 + _i * 64 + 0])
                        acc[0] += fs[(tid % 32) * 2 + _i * 64 + 1] * T.Cast("float32", sm[(_t % 2) * 21504 + (tid // 32) * 2688 + (tid % 32) * 2 + _i * 64 + 1])
                    red[tid] = acc[0]
                    T.sync_threads()
                    if tid < 8:
                        acc[0] = 0.0
                        for _q in T.serial(32):
                            acc[0] += red[tid * 32 + _q]
                        if T.max(T.min(cta * 79 + (_t) * 8, 10296), 0) + tid >= cta * 79:
                            out[T.max(T.min(cta * 79 + (_t) * 8, 10296), 0) + tid] = acc[0]
                    T.sync_threads()
                    if _t + 2 < 10:
                        T.tma_copy(w[T.max(T.min(sel[_rep] * 27697152 + T.max(T.min(cta * 79 + (_t + 2) * 8, 10296), 0) * 2688, 637012992), 0):T.max(T.min(sel[_rep] * 27697152 + T.max(T.min(cta * 79 + (_t + 2) * 8, 10296), 0) * 2688, 637012992), 0) + 21504], sm[(_t % 2) * 21504:(_t % 2) * 21504 + 21504], barrier=bar[_t % 2])
                        if T.shuffle_elect(THREADS):
                            T.barrier_arrive(bar[_t % 2])
                if T.shuffle_elect(THREADS):
                    T.barrier_arrive(bar[0])
                T.mbarrier_wait_parity(bar[0], 1)
                if T.shuffle_elect(THREADS):
                    T.barrier_arrive(bar[1])
                T.mbarrier_wait_parity(bar[1], 1)
    return main