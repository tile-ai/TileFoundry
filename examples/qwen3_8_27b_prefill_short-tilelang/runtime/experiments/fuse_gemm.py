"""Step-3 fusion experiments around the GEMMs, GPU time at each length:
  residual: GEMM with residual epilogue  vs  plain GEMM + separate add kernel
  swiglu:   two-source GEMM with SwiGLU epilogue  vs  packed [gate;up] GEMM + separate silu*mul kernel
            vs  two plain GEMMs + separate silu*mul kernel
  in_proj:  packed [qkv|z|b|a] GEMM  vs  three separate GEMMs (qkv, z, b|a)"""
import json, sys, torch
sys.path.insert(0, '.')
import tilelang
import tilelang.language as T
from runtime import plan
from runtime.kernels.gemm import gemm_kernel, splitk_reduce_kernel, GemmConfig
from runtime.benchutil import graph_time_us

def ew_add(M, N, block=256, per=8):
    dt, acc = "bfloat16", "float"
    @T.prim_func
    def main(R: T.Tensor((M, N), dt), X: T.Tensor((M, N), dt), C: T.Tensor((M, N), dt)):
        with T.Kernel(T.ceildiv(M * N, block * per), threads=block) as b:
            for i in T.Parallel(block * per):
                idx = b * block * per + i
                C[idx // N, idx % N] = T.cast(T.cast(R[idx // N, idx % N], acc) + T.cast(X[idx // N, idx % N], acc), dt)
    return tilelang.compile(main, target='cuda')

def ew_silu_mul(M, N, packed, block=256, per=8):
    dt, acc = "bfloat16", "float"
    @T.prim_func
    def main(G: T.Tensor((M, 2 * N if packed else N), dt), U: T.Tensor((M, N), dt), O: T.Tensor((M, N), dt)):
        with T.Kernel(T.ceildiv(M * N, block * per), threads=block) as b:
            for i in T.Parallel(block * per):
                idx = b * block * per + i
                m = idx // N
                n = idx % N
                g = T.cast(G[m, n], acc)
                u = T.cast(G[m, N + n], acc) if packed else T.cast(U[m, n], acc)
                s = T.cast(g * T.fast_rcp(T.float32(1.0) + T.exp(-g)), dt)
                O[m, n] = T.cast(T.cast(s, acc) * u, dt)
    return tilelang.compile(main, target='cuda')

res = {}
for M in (128, 256, 512, 1024):
    p = plan.for_length(M)
    big = GemmConfig(BN=128, stages=6, swizzle=0) if M == 128 else GemmConfig(BN=256, stages=4)
    # residual (down: K=17408)
    K, N = 17408, 5120
    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16); W = (torch.randn(N, K, device='cuda') * 0.01).bfloat16()
    R = torch.randn(M, N, device='cuda', dtype=torch.bfloat16); C = torch.empty_like(R); X = torch.empty_like(R)
    cfg = p.gemm['down']
    if cfg.split_k > 1:
        kf = gemm_kernel(M, N, K, cfg, 'splitk'); red = splitk_reduce_kernel(M, N, cfg.split_k, True); P = torch.empty(cfg.split_k, M, N, device='cuda')
        fused = lambda: (kf(A, W, P), red(P, R, C))
        red0 = splitk_reduce_kernel(M, N, cfg.split_k, False); kadd = ew_add(M, N)
        unf = lambda: (kf(A, W, P), red0(P, X), kadd(R, X, C))
    else:
        kf = gemm_kernel(M, N, K, cfg, 'residual'); ks = gemm_kernel(M, N, K, cfg, 'store'); kadd = ew_add(M, N)
        fused = lambda: kf(A, W, R, C)
        unf = lambda: (ks(A, W, X), kadd(R, X, C))
    res[f'residual_down_T{M}'] = (graph_time_us(fused), graph_time_us(unf))
    # swiglu
    K, N = 5120, 17408
    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    Wg = (torch.randn(N, K, device='cuda') * 0.02).bfloat16(); Wu = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    Wp = torch.cat([Wg, Wu]).contiguous()
    O = torch.empty(M, N, device='cuda', dtype=torch.bfloat16); G2 = torch.empty(M, 2 * N, device='cuda', dtype=torch.bfloat16)
    G = torch.empty(M, N, device='cuda', dtype=torch.bfloat16); U = torch.empty_like(G)
    kf = gemm_kernel(M, N, K, p.gemm['gate_up'], 'swiglu2')
    kp = gemm_kernel(M, 2 * N, K, big, 'store'); kg = gemm_kernel(M, N, K, big, 'store')
    sm_p = ew_silu_mul(M, N, True); sm_s = ew_silu_mul(M, N, False)
    res[f'swiglu_T{M}'] = (graph_time_us(lambda: kf(A, Wg, Wu, O)), graph_time_us(lambda: (kp(A, Wp, G2), sm_p(G2, U, O))),
                          graph_time_us(lambda: (kg(A, Wg, G), kg(A, Wu, U), sm_s(G, U, O))))
    del Wp
    # in_proj (GDN): packed 16640 vs qkv 10240 + z 6144 + ba 96
    Wq = (torch.randn(16640, K, device='cuda') * 0.02).bfloat16(); Cq = torch.empty(M, 16640, device='cuda', dtype=torch.bfloat16)
    kpk = gemm_kernel(M, 16640, K, p.gemm['gdn_in'], 'store')
    k1 = gemm_kernel(M, 10240, K, big, 'store'); k2 = gemm_kernel(M, 6144, K, big, 'store')
    k3 = gemm_kernel(M, 96, K, GemmConfig(BN=96, stages=4, swizzle=0), 'store')
    W1, W2, W3 = Wq[:10240], Wq[10240:16384], Wq[16384:16480]
    C1 = torch.empty(M, 10240, device='cuda', dtype=torch.bfloat16); C2 = torch.empty(M, 6144, device='cuda', dtype=torch.bfloat16); C3 = torch.empty(M, 96, device='cuda', dtype=torch.bfloat16)
    res[f'in_proj_T{M}'] = (graph_time_us(lambda: kpk(A, Wq, Cq)), graph_time_us(lambda: (k1(A, W1, C1), k2(A, W2, C2), k3(A, W3, C3))))
    for key in [k for k in res if k.endswith(f'T{M}')]:
        print(key, ' | '.join(f'{v:.1f}us' for v in res[key]), flush=True)
json.dump(res, open('reports/fuse_gemm.json', 'w'), indent=1)
