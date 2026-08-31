"""What the linear shared layout costs the tensor cores.

The K/V block arrives by descriptorless bulk copy, which writes linear bytes, so
the buffer the gemm reads cannot carry the swizzle the gemm would pick for
itself.  This is the same four products a decode block does -- two KV groups,
scores and context each -- with the layout pinned and with it inferred.
"""
import sys
import time
import torch
import tilelang
import tilelang.language as T

GQA, ABLK, DH, HKV = 16, 128, 128, 2
KVROW = HKV * DH
THREADS = 256
REPS = 200


@tilelang.jit
def build(linear: bool, check: bool = False):
    @T.prim_func
    def main(q: T.Tensor((HKV * GQA, DH), "bfloat16"),
             kv: T.Tensor((ABLK, KVROW), "bfloat16"),
             vv: T.Tensor((ABLK, KVROW), "bfloat16"),
             o_out: T.Tensor((HKV * GQA, DH), "float32")):
        with T.Kernel(1 if check else 132, threads=THREADS) as bx:
            qs = T.alloc_shared((HKV * GQA, DH), "bfloat16")
            ks = T.alloc_shared((ABLK, KVROW), "bfloat16")
            vs = T.alloc_shared((ABLK, KVROW), "bfloat16")
            ps = T.alloc_shared((GQA, ABLK), "bfloat16")
            sf = T.alloc_fragment((GQA, ABLK), "float32")
            of0 = T.alloc_fragment((GQA, DH), "float32")
            of1 = T.alloc_fragment((GQA, DH), "float32")
            if linear:
                T.annotate_layout({
                    ks: T.Layout((ABLK, KVROW), lambda i, j: i * KVROW + j),
                    vs: T.Layout((ABLK, KVROW), lambda i, j: i * KVROW + j)})
            T.copy(q, qs)
            T.copy(kv, ks)
            T.copy(vv, vs)
            T.clear(of0)
            T.clear(of1)
            for _ in T.serial(1 if check else REPS):
                T.clear(sf)
                T.gemm(qs[0:GQA, :], ks[:, 0:DH], sf, transpose_B=True)
                T.copy(sf, ps)
                T.gemm(ps, vs[:, 0:DH], of0)
                T.clear(sf)
                T.gemm(qs[GQA:2 * GQA, :], ks[:, DH:2 * DH], sf, transpose_B=True)
                T.copy(sf, ps)
                T.gemm(ps, vs[:, DH:2 * DH], of1)
            T.copy(of0, o_out[0:GQA, :])
            T.copy(of1, o_out[GQA:2 * GQA, :])
    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(HKV * GQA, DH, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(ABLK, KVROW, device="cuda", dtype=torch.bfloat16)
    vv = torch.randn(ABLK, KVROW, device="cuda", dtype=torch.bfloat16)
    ref = torch.cat([
        (q[g * GQA:(g + 1) * GQA].float() @ kv[:, g * DH:(g + 1) * DH].float().t())
        .to(torch.bfloat16).float() @ vv[:, g * DH:(g + 1) * DH].float()
        for g in range(HKV)])
    for linear in (True, False):
        tag = "linear  " if linear else "swizzled"
        o = torch.zeros(HKV * GQA, DH, device="cuda", dtype=torch.float32)
        build(linear, check=True)(q, kv, vv, o)
        torch.cuda.synchronize()
        rel = ((o - ref).norm() / ref.norm()).item()
        fn = build(linear)
        for _ in range(3):
            fn(q, kv, vv, o)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 20
        for _ in range(N):
            fn(q, kv, vv, o)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N
        blocks = 132 * REPS
        mac = blocks * HKV * 2 * GQA * ABLK * DH
        print(f"{tag}  rel_l2 {rel:.3e}   {dt / blocks * 1e9:6.1f} ns per block"
              f"   {2 * mac / dt / 1e12:6.1f} TFLOP/s", flush=True)
