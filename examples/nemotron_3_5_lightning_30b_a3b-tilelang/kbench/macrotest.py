import torch, tilelang
import tilelang.language as T

def PY(n):
    """A trace-time range: `range` itself is overridden to a device loop."""
    return tuple(range(n))


@T.macro
def addrows(A, B, n: int, k: int, tid, off: int):
    for j in PY(n):
        if j % 2 == 0:
            B[off + j] = A[j * k + tid] + float(j)
        else:
            B[off + j] = A[j * k + tid] - float(j)


@tilelang.jit(out_idx=[-1])
def build():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=8) as bx:
            tid = T.get_thread_binding()
            addrows(A, B, 4, 8, tid, 0)
    return main


if __name__ == "__main__":
    fn = build()
    a = torch.arange(64, device="cuda", dtype=torch.float32)
    out = fn(a)
    print(out[:8].tolist())
    print("want", [0.0, 8-1, 16+2, 24-3, 0, 0, 0, 0])
