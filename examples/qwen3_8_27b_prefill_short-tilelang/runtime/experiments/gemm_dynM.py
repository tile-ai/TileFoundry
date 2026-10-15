"""Dims experiment: one GEMM kernel with a dynamic M (T.dynamic) serving every length vs
a kernel compiled for each static M. Same tile config (128x256x64 s4 sw8, FullRow)."""
import sys, torch
sys.path.insert(0, '.')
import tilelang
import tilelang.language as T
from runtime.benchutil import graph_time_us
from runtime.kernels.gemm import gemm_kernel, GemmConfig
POL = T.GemmWarpPolicy.FullRow

def make_dyn(N, K, BM=128, BN=256, BK=64, stages=4):
    dt, acc = "bfloat16", "float"
    M = T.dynamic("M")
    @T.prim_func
    def main(A: T.Tensor((M, K), dt), W: T.Tensor((N, K), dt), C: T.Tensor((M, N), dt)):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=256) as (bx, by):
            As = T.alloc_shared((BM, BK), dt); Ws = T.alloc_shared((BN, BK), dt)
            Cl = T.alloc_fragment((BM, BN), acc); Cs = T.alloc_shared((BM, BN), dt)
            T.use_swizzle(panel_size=8)
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[by * BM, k * BK], As)
                T.copy(W[bx * BN, k * BK], Ws)
                T.gemm(As, Ws, Cl, transpose_B=True, policy=POL)
            T.copy(Cl, Cs)
            T.copy(Cs, C[by * BM, bx * BN])
    return tilelang.compile(main, target='cuda')

N, K = 16640, 5120
kd = make_dyn(N, K)
W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
for M in (128, 256, 512, 1024):
    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16); C = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    ks = gemm_kernel(M, N, K, GemmConfig(BN=256, stages=4), 'store')
    kd(A, W, C); torch.cuda.synchronize(); okd = torch.equal(C, A @ W.t())
    td = graph_time_us(lambda: kd(A, W, C)); ts = graph_time_us(lambda: ks(A, W, C))
    print(f"M={M}: dynamic-M {td:.1f}us (bitwise {okd}) | static-M {ts:.1f}us | dynamic/static {td/ts:.3f}", flush=True)
