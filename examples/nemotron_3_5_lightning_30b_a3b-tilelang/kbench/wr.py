import tilelang, tilelang.language as T, torch
@tilelang.jit
def build():
    @T.prim_func
    def main(x: T.Tensor((256,), "float32"), out: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=256) as bx:
            acc = T.alloc_local((1,), "float32")
            tid = T.get_thread_binding()
            acc[0] = x[tid]
            acc[0] = T.warp_reduce_sum(acc[0])
            if tid % 32 == 0:
                out[tid // 32] = acc[0]
    return main
src = build().get_kernel_source()
i = src.index("main_kernel")
print(src[i - 100:i + 1600])
