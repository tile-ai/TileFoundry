"""Which shapes of TMA address keep the one-dimensional bulk path?

The failing case reports `Descriptor-based TMA cannot use global base pointer`,
which is the lowering having given up on the bulk path and reached for a
TensorMap descriptor -- and a descriptor cannot be built for a pointer the
kernel read out of an address table. This asks, one address shape at a time,
which ones survive.
"""
import sys
import torch
import tilelang
import tilelang.language as T

N, THREADS = 4096, 128
FORM = sys.argv[1] if len(sys.argv) > 1 else "bx_mul"


def build(form):
    src = f'''
import tilelang, tilelang.language as T
N, THREADS = {N}, {THREADS}


@tilelang.jit
def build():
    @T.prim_func
    def main(ptrs: T.Tensor((4,), "int64"), stride: T.Tensor((1,), "int32"),
             out: T.Tensor((8,), "float32")):
        with T.Kernel({"8" if form.startswith("bx") else "8, 4"}, threads=THREADS) as {"bx" if form.startswith("bx") else "(bx, by)"}:
            smem = T.alloc_shared((N,), "bfloat16")
            bar = T.alloc_barrier([1])
            tid = T.get_thread_binding()
            src = T.make_tensor_from_addr(ptrs[0], (1 << 20,), "bfloat16")
            st = T.Cast("int32", stride[0])
            for _b in T.serial(2):
                T.tma_copy(src[{{OFF}}:{{OFF}} + N], smem[0:N], barrier=bar[0])
                if T.shuffle_elect(THREADS):
                    T.barrier_arrive(bar[0])
                T.mbarrier_wait_parity(bar[0], _b % 2)
                if tid < 8:
                    out[tid] = T.Cast("float32", smem[tid])
    return main
'''
    offs = {
        "none": "T.max(T.min(_b * N, (1 << 20) - N), 0)",
        "bx_mul": "T.max(T.min(bx * st + _b * N, (1 << 20) - N), 0)",
        "bx_div": "T.max(T.min(bx // 2 + _b * N, (1 << 20) - N), 0)",
        "bx_mod": "T.max(T.min(bx % 2 * N + _b * N, (1 << 20) - N), 0)",
        "2d_by_mul": "T.max(T.min(by * st + _b * N, (1 << 20) - N), 0)",
        "2d_bx_by": "T.max(T.min(bx * st + by * N + _b * N, (1 << 20) - N), 0)",
        "2d_by_const": "T.max(T.min((by + 4 * _b) * N, (1 << 20) - N), 0)",
        "2d_by_constx": "T.max(T.min(by * N + _b * N, (1 << 20) - N), 0)",
        "2d_by_st_only": "T.max(T.min(by * st, (1 << 20) - N), 0)",
    }
    import pathlib, importlib
    tag = f"_t2d_{form}"
    pathlib.Path(f"kbench/{tag}.py").write_text(src.replace("{OFF}", offs[form]))
    return importlib.import_module(f"kbench.{tag}").build()


if __name__ == "__main__":
    for form in ("none", "bx_mul", "bx_div", "bx_mod", "2d_by_const",
                 "2d_by_constx", "2d_by_st_only", "2d_by_mul", "2d_bx_by"):
        try:
            build(form)
            print(f"{form:<12} BULK OK")
        except Exception as error:
            msg = str(error).splitlines()[-1][:80]
            print(f"{form:<12} FAIL  {msg}")
