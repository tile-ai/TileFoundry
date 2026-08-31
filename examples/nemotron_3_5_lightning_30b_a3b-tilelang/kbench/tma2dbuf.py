"""Can one shared buffer be both a bulk-copy destination and a tensor-core operand?

`T.gemm` wants a two-dimensional buffer it may swizzle; a descriptorless bulk
copy wants a rank-1 destination and writes linear bytes.  The buffer is declared
two-dimensional so the gemm will take it, pinned to a linear layout so the bytes
mean what the copy wrote, and the copy addresses a rank-1 view of it.
"""
import torch
import tilelang
import tilelang.language as T

ABLK, KVROW, DH, M, THREADS = 128, 256, 128, 16, 256
NEL = ABLK * KVROW


@tilelang.jit
def build(linear: bool):
    @T.prim_func
    def main(ptrs: T.Tensor((4,), "int64"), stride: T.Tensor((1,), "int32"),
             out: T.Tensor((M, DH), "float32")):
        with T.Kernel(4, 4, threads=THREADS) as (bx, by):
            kvb = T.alloc_shared((ABLK, KVROW), "bfloat16")
            kvf = T.view(kvb, (NEL,))
            qs = T.alloc_shared((M, DH), "bfloat16")
            ss = T.alloc_shared((M, DH), "float32")
            sf = T.alloc_fragment((M, DH), "float32")
            if linear:
                T.annotate_layout({kvb: T.Layout((ABLK, KVROW),
                                                 lambda i, j: i * KVROW + j)})
            bar = T.alloc_barrier([1])
            tid = T.get_thread_binding()
            src = T.make_tensor_from_addr(ptrs[0], (1 << 22,), "bfloat16")
            st = T.Cast("int32", stride[0])
            lo = T.max(T.min(by * st, (1 << 22) - NEL), 0)
            T.tma_copy(src[lo:lo + NEL], kvf, barrier=bar[0])
            if T.shuffle_elect(THREADS):
                T.barrier_arrive(bar[0])
            T.mbarrier_wait_parity(bar[0], 0)
            for i, j in T.Parallel(M, DH):
                qs[i, j] = T.Cast("bfloat16", 1.0 if j == 0 else 0.0)
            T.clear(sf)
            T.gemm(qs, kvb[:, 0:DH], sf, transpose_B=True)
            T.copy(sf, ss)
            if bx == 0 and by == 0:
                for j in T.serial(DH):
                    if tid == 0:
                        out[0, j] = ss[0, j]
    return main


if __name__ == "__main__":
    buf = torch.arange(1 << 22, device="cuda", dtype=torch.bfloat16)
    ptrs = torch.tensor([buf.data_ptr()] * 4, device="cuda", dtype=torch.int64)
    st = torch.tensor([NEL], device="cuda", dtype=torch.int32)
    # q is one-hot on column 0, so row j of the product is K[j, 0].
    want = buf[:NEL].view(ABLK, KVROW)[:DH, 0].float()
    for linear in (True, False):
        out = torch.zeros(M, DH, device="cuda", dtype=torch.float32)
        tag = "linear" if linear else "swizzled"
        try:
            build(linear)(ptrs, st, out)
            torch.cuda.synchronize()
            ok = torch.allclose(out[0], want)
            print(f"{tag:9s} {'OK   ' if ok else 'WRONG'} got {out[0, :4].tolist()}"
                  f" want {want[:4].tolist()}")
        except Exception as error:
            print(f"{tag:9s} FAIL  " + str(error).strip().splitlines()[0][:160])
