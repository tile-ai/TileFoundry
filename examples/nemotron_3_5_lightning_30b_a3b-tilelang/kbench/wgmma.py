"""The two attention GEMMs of a decode step, on the tensor cores.

Per KV group and per 128-position block a decode step wants exactly two
products, and both are the same shape:

    scores   Q(16 heads x 128 dims) @ K^T(128 dims x 128 positions) -> (16, 128)
    context  P(16 heads x 128 positions) @ V(128 positions x 128 dims) -> (16, 128)

M is 16 because GQA puts 16 query heads on one KV head, and one decode step has
one query token. WGMMA's M is 64, so three quarters of the tensor core goes
unused -- which is what every GQA decode kernel pays -- and it is still an
enormous multiple of doing it a multiply-add at a time on the CUDA cores.
"""
import sys
import time
import torch
import tilelang
import tilelang.language as T

HEADS, POS, DH = 16, 128, 128
THREADS = 256
REPS = 200


@tilelang.jit
def build():
    @T.prim_func
    def main(q: T.Tensor((HEADS, DH), "bfloat16"), k: T.Tensor((POS, DH), "bfloat16"),
             v: T.Tensor((POS, DH), "bfloat16"), s_out: T.Tensor((HEADS, POS), "float32"),
             o_out: T.Tensor((HEADS, DH), "float32")):
        with T.Kernel(1, threads=THREADS) as bx:
            qs = T.alloc_shared((HEADS, DH), "bfloat16")
            ks = T.alloc_shared((POS, DH), "bfloat16")
            vs = T.alloc_shared((POS, DH), "bfloat16")
            ps = T.alloc_shared((HEADS, POS), "bfloat16")
            sf = T.alloc_fragment((HEADS, POS), "float32")
            of = T.alloc_fragment((HEADS, DH), "float32")
            T.copy(q, qs)
            T.copy(k, ks)
            T.copy(v, vs)
            T.clear(sf)
            # scores: K is stored (position, dim), so the contraction is over its
            # second axis -- transpose_B, no rearrangement in memory.
            T.gemm(qs, ks, sf, transpose_B=True)
            T.copy(sf, s_out)
            T.copy(sf, ps)
            T.clear(of)
            # context: V is stored (position, dim) and the contraction is over
            # position, which is its first axis -- no transpose.
            T.gemm(ps, vs, of)
            T.copy(of, o_out)
    return main


@tilelang.jit
def build_timed():
    @T.prim_func
    def main(q: T.Tensor((HEADS, DH), "bfloat16"), k: T.Tensor((POS, DH), "bfloat16"),
             v: T.Tensor((POS, DH), "bfloat16"), o_out: T.Tensor((HEADS, DH), "float32")):
        with T.Kernel(132, threads=THREADS) as bx:
            qs = T.alloc_shared((HEADS, DH), "bfloat16")
            ks = T.alloc_shared((POS, DH), "bfloat16")
            vs = T.alloc_shared((POS, DH), "bfloat16")
            ps = T.alloc_shared((HEADS, POS), "bfloat16")
            sf = T.alloc_fragment((HEADS, POS), "float32")
            of = T.alloc_fragment((HEADS, DH), "float32")
            T.copy(q, qs)
            T.copy(k, ks)
            T.copy(v, vs)
            T.clear(of)
            for _ in T.serial(REPS):
                T.clear(sf)
                T.gemm(qs, ks, sf, transpose_B=True)
                T.copy(sf, ps)
                T.gemm(ps, vs, of)
            T.copy(of, o_out)
    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(HEADS, DH, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(POS, DH, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(POS, DH, device="cuda", dtype=torch.bfloat16)
    s = torch.zeros(HEADS, POS, device="cuda", dtype=torch.float32)
    o = torch.zeros(HEADS, DH, device="cuda", dtype=torch.float32)
    build()(q, k, v, s, o)
    torch.cuda.synchronize()
    s_ref = (q.float() @ k.float().t())
    o_ref = s.to(torch.bfloat16).float() @ v.float()
    print(f"scores  rel_l2 {((s - s_ref).norm() / s_ref.norm()).item():.3e}")
    print(f"context rel_l2 {((o - o_ref).norm() / o_ref.norm()).item():.3e}")

    fn = build_timed()
    for _ in range(3):
        fn(q, k, v, o)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 20
    for _ in range(N):
        fn(q, k, v, o)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N
    pairs = 132 * REPS
    mac = pairs * 2 * HEADS * POS * DH
    print(f"\n{pairs} (scores+context) pairs in {dt * 1e6:.1f} us"
          f"   {dt / pairs * 1e9:.1f} ns per pair"
          f"   {2 * mac / dt / 1e12:.1f} TFLOP/s")
