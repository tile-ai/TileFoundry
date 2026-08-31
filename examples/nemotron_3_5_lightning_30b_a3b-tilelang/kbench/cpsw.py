"""Can a plain copy from a table pointer land in a buffer the gemm swizzled?

`T.tma_copy` cannot: a tiled TMA is a host-encoded descriptor and the source is
a pointer the kernel read at run time.  The error says as much and points at
`T.copy`, which is allowed a descriptorless fallback -- so the question is
whether that fallback survives the swizzle the tensor cores want.
"""
import torch
import tilelang
import tilelang.language as T

ABLK, KVROW, DH, GQA, THREADS = 128, 256, 128, 16, 256
NEL = ABLK * KVROW


@tilelang.jit
def build(instr):
    kw = {} if instr is None else {"prefer_instruction": instr}

    @T.prim_func
    def main(ptrs: T.Tensor((2,), "int64"), stride: T.Tensor((1,), "int32"),
             q: T.Tensor((GQA, DH), "float32"), o: T.Tensor((GQA, DH), "float32")):
        with T.Kernel(4, 4, threads=THREADS) as (bx, by):
            ks = T.alloc_shared((ABLK, KVROW), "bfloat16")
            qs = T.alloc_shared((GQA, DH), "bfloat16")
            fo = T.alloc_shared((GQA, DH), "float32")
            sf = T.alloc_fragment((GQA, DH), "float32")
            tid = T.get_thread_binding()
            nb = T.Cast("int32", stride[0])
            src = T.make_tensor_from_addr(ptrs[0], (1 << 17, ABLK, KVROW), "bfloat16")
            for i, j in T.Parallel(GQA, DH):
                qs[i, j] = T.Cast("bfloat16", q[i, j])
            T.copy(src[T.max(T.min(by, nb - 1), 0), :, :], ks, **kw)
            T.sync_threads()
            T.clear(sf)
            T.gemm(qs, ks[:, 0:DH], sf, transpose_B=True)
            T.copy(sf, fo)
            T.sync_threads()
            if bx == 0 and by == 0:
                for i in T.serial(GQA):
                    if tid < DH:
                        o[i, tid] = fo[i, tid]
    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    nb = 64
    buf = torch.randn(nb, ABLK, KVROW, device="cuda", dtype=torch.bfloat16)
    ptrs = torch.tensor([buf.data_ptr()] * 2, device="cuda", dtype=torch.int64)
    st = torch.tensor([nb], device="cuda", dtype=torch.int32)
    q = torch.randn(GQA, DH, device="cuda", dtype=torch.float32)
    ref = q.to(torch.bfloat16).float() @ buf[0, :, :DH].float().t()
    for instr in (None, "tma", "cp_async", "sync"):
        o = torch.zeros(GQA, DH, device="cuda", dtype=torch.float32)
        try:
            build(instr)(ptrs, st, q, o)
            torch.cuda.synchronize()
            rel = ((o - ref).norm() / ref.norm()).item()
            print(f"{str(instr):9s} {'OK   ' if rel < 3e-3 else 'WRONG'} rel_l2 {rel:.3e}")
        except Exception as error:
            print(f"{str(instr):9s} FAIL  " + str(error).strip().splitlines()[0][:150])
