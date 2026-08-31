"""The arena as one two-dimensional buffer: bulk-copy destination and gemm operand.

Everything the mega kernel keeps in shared already lives in one flat arena that
the streaming stages bulk-copy into.  Declaring that arena two-dimensional and
pinning it to a linear layout costs nothing -- the flat view still takes the
bulk copies, byte for byte where they were -- and buys the attention block its
tensor-core operands as plain row and column slices of the same bytes.
"""
import torch
import tilelang
import tilelang.language as T

HQ, HKV, GQA, DH, ABLK = 32, 2, 16, 128, 128
KVROW = HKV * DH
SMROW, ARENA = 352, 352 * 256
FROW, FARENA = 64, 64 * 128
KROW, VROW, QROW, PROW = 0, 128, 256, 288
THREADS = 256


@tilelang.jit
def build():
    @T.prim_func
    def main(ptrs: T.Tensor((2,), "int64"), stride: T.Tensor((1,), "int32"),
             q: T.Tensor((HQ, DH), "float32"), o: T.Tensor((HQ, DH), "float32")):
        with T.Kernel(4, 4, threads=THREADS) as (bx, by):
            sm = T.alloc_shared((SMROW, KVROW), "bfloat16")
            fs = T.alloc_shared((FROW, DH), "float32")
            T.annotate_layout({
                sm: T.Layout((SMROW, KVROW), lambda i, j: i * KVROW + j),
                fs: T.Layout((FROW, DH), lambda i, j: i * DH + j)})
            smf = T.view(sm, (ARENA,))
            sf = T.alloc_fragment((GQA, ABLK), "float32")
            of = T.alloc_fragment((GQA, DH), "float32")
            bar = T.alloc_barrier([1] * 2)
            tid = T.get_thread_binding()
            src = T.make_tensor_from_addr(ptrs[0], (1 << 24,), "bfloat16")
            st = T.Cast("int32", stride[0])
            # K and V, one bulk copy each, into the flat view of the arena.
            lok = T.max(T.min(by * st, (1 << 24) - ABLK * KVROW), 0)
            T.tma_copy(src[lok:lok + ABLK * KVROW],
                       smf[KROW * KVROW:(KROW + ABLK) * KVROW], barrier=bar[0])
            if T.shuffle_elect(THREADS):
                T.barrier_arrive(bar[0])
            lov = T.max(T.min(by * st + ABLK * KVROW,
                              (1 << 24) - ABLK * KVROW), 0)
            T.tma_copy(src[lov:lov + ABLK * KVROW],
                       smf[VROW * KVROW:(VROW + ABLK) * KVROW], barrier=bar[1])
            if T.shuffle_elect(THREADS):
                T.barrier_arrive(bar[1])
            for i, j in T.Parallel(HQ, DH):
                sm[QROW + i, j] = T.Cast("bfloat16", q[i, j])
            T.mbarrier_wait_parity(bar[0], 0)
            T.sync_threads()
            T.clear(sf)
            T.gemm(sm[QROW:QROW + GQA, 0:DH], sm[KROW:KROW + ABLK, 0:DH],
                   sf, transpose_B=True)
            T.copy(sf, fs[0:GQA, :])
            T.sync_threads()
            for i, j in T.Parallel(GQA, ABLK):
                sm[PROW + i, j] = T.Cast("bfloat16", fs[i, j])
            T.mbarrier_wait_parity(bar[1], 0)
            T.sync_threads()
            T.clear(of)
            T.gemm(sm[PROW:PROW + GQA, 0:ABLK], sm[VROW:VROW + ABLK, 0:DH], of)
            T.copy(of, fs[GQA:2 * GQA, :])
            T.sync_threads()
            if bx == 0 and by == 0:
                for i in T.serial(GQA):
                    if tid < DH:
                        o[i, tid] = fs[GQA + i, tid]
    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 1 << 24
    buf = (torch.rand(n, device="cuda", dtype=torch.float32) - 0.5).to(torch.bfloat16)
    ptrs = torch.tensor([buf.data_ptr()] * 2, device="cuda", dtype=torch.int64)
    st = torch.tensor([2 * ABLK * KVROW], device="cuda", dtype=torch.int32)
    q = torch.randn(HQ, DH, device="cuda", dtype=torch.float32)
    o = torch.zeros(HQ, DH, device="cuda", dtype=torch.float32)
    try:
        build()(ptrs, st, q, o)
        torch.cuda.synchronize()
        k = buf[:ABLK * KVROW].view(ABLK, KVROW)[:, :DH].float()
        v = buf[ABLK * KVROW:2 * ABLK * KVROW].view(ABLK, KVROW)[:, :DH].float()
        s = q[:GQA].to(torch.bfloat16).float() @ k.t()
        ref = s.to(torch.bfloat16).float() @ v
        rel = ((o[:GQA] - ref).norm() / ref.norm()).item()
        print(f"arena2d {'OK  ' if rel < 3e-3 else 'WRONG'} rel_l2 {rel:.3e}")
    except Exception as error:
        print("arena2d FAIL: " + str(error).strip().splitlines()[0][:200])
