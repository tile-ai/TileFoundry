"""GEMM kernels for the prefill's linears: C[M, N] = A[M, K] @ W[N, K]^T.

A is the activation (row-major, K contiguous), W is the checkpoint's own (out, in)
weight (K contiguous), so both operands are K-major, which is what WGMMA reads
from shared memory without a transpose. bf16 in, f32 accumulate.

Epilogues (each one is a rounding contract taken from model.py):
  store     C = bf16(acc)
  residual  C = bf16(R + bf16(acc))                       (attended = hidden + mixed)
  swiglu    W rows interleaved [gate 64 | up 64 | gate 64 | up 64 ...]; output
            O[:, j] = bf16(bf16(silu(bf16(g))) * bf16(u))    (tf.silu(gate) * up)
  swiglu2   the same from two unpacked weights (w_gate, w_up): BN is one half's
            width; two B rings, two WGMMA chains, epilogue on the accumulators
  splitk    P[s, M, N] = f32 partial over the s-th K slice (a reduce kernel finishes it)

Every kernel is built for one static (M, N, K) and one tile config and cached.
"""

from dataclasses import dataclass
from functools import lru_cache

import tilelang
import tilelang.language as T

GU_BLOCK = 64  # gate/up interleave granularity of the packed w_gate/w_up weight
# Warpgroup g of the two consumer warpgroups owns rows [64g, 64g+64) of the tile, as
# the finalized TIR states (tir/gemm/*: acc layout (2 @ group, ...) on the M axis).
_POLICY = T.GemmWarpPolicy.FullRow


@dataclass(frozen=True)
class GemmConfig:
    BM: int = 128
    BN: int = 256
    BK: int = 64
    stages: int = 3
    threads: int = 256
    swizzle: int = 8  # rasterization panel (0 = none)
    split_k: int = 1

    def tag(self) -> str:
        sk = f"_sk{self.split_k}" if self.split_k > 1 else ""
        return f"{self.BM}x{self.BN}x{self.BK}_s{self.stages}_t{self.threads}_sw{self.swizzle}{sk}"


def _compile(func, name=None):
    if name:
        func = func.with_attr("global_symbol", name)
    return tilelang.compile(func, target="cuda")


@lru_cache(maxsize=None)
def gemm_kernel(M: int, N: int, K: int, cfg: GemmConfig, epilogue: str = "store"):
    BM, BN, BK = cfg.BM, cfg.BN, cfg.BK
    dt, acc = "bfloat16", "float"
    if epilogue == "swiglu":
        assert BN % (2 * GU_BLOCK) == 0 and N % (2 * GU_BLOCK) == 0
        NO = N // 2
    k_tiles = T.ceildiv(K, BK)

    if epilogue == "store":

        @T.prim_func
        def main(A: T.Tensor((M, K), dt), W: T.Tensor((N, K), dt), C: T.Tensor((M, N), dt)):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=cfg.threads) as (bx, by):
                As = T.alloc_shared((BM, BK), dt)
                Ws = T.alloc_shared((BN, BK), dt)
                Cl = T.alloc_fragment((BM, BN), acc)
                Cs = T.alloc_shared((BM, BN), dt)
                if cfg.swizzle:
                    T.use_swizzle(panel_size=cfg.swizzle)
                T.clear(Cl)
                for k in T.Pipelined(k_tiles, num_stages=cfg.stages):
                    T.copy(A[by * BM, k * BK], As)
                    T.copy(W[bx * BN, k * BK], Ws)
                    T.gemm(As, Ws, Cl, transpose_B=True, policy=_POLICY)
                T.copy(Cl, Cs)
                T.copy(Cs, C[by * BM, bx * BN])

    elif epilogue == "residual":

        @T.prim_func
        def main(A: T.Tensor((M, K), dt), W: T.Tensor((N, K), dt), R: T.Tensor((M, N), dt), C: T.Tensor((M, N), dt)):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=cfg.threads) as (bx, by):
                As = T.alloc_shared((BM, BK), dt)
                Ws = T.alloc_shared((BN, BK), dt)
                Cl = T.alloc_fragment((BM, BN), acc)
                Rl = T.alloc_fragment((BM, BN), dt)
                Ol = T.alloc_fragment((BM, BN), dt)
                Cs = T.alloc_shared((BM, BN), dt)
                if cfg.swizzle:
                    T.use_swizzle(panel_size=cfg.swizzle)
                T.clear(Cl)
                for k in T.Pipelined(k_tiles, num_stages=cfg.stages):
                    T.copy(A[by * BM, k * BK], As)
                    T.copy(W[bx * BN, k * BK], Ws)
                    T.gemm(As, Ws, Cl, transpose_B=True, policy=_POLICY)
                # TIR: mixed = cast(acc, bf16); r_tile = copy(residual tile) into the
                # accumulator's register layout; tile_out = cast(f32(r) + f32(mixed), bf16).
                T.copy(R[by * BM, bx * BN], Rl)
                for i, j in T.Parallel(BM, BN):
                    Ol[i, j] = T.cast(T.cast(Rl[i, j], acc) + T.cast(T.cast(Cl[i, j], dt), acc), dt)
                T.copy(Ol, Cs)
                T.copy(Cs, C[by * BM, bx * BN])

    elif epilogue == "swiglu":

        @T.prim_func
        def main(A: T.Tensor((M, K), dt), W: T.Tensor((N, K), dt), O: T.Tensor((M, NO), dt)):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=cfg.threads) as (bx, by):
                As = T.alloc_shared((BM, BK), dt)
                Ws = T.alloc_shared((BN, BK), dt)
                Cl = T.alloc_fragment((BM, BN), acc)
                Cs = T.alloc_shared((BM, BN), dt)
                Os = T.alloc_shared((BM, BN // 2), dt)
                if cfg.swizzle:
                    T.use_swizzle(panel_size=cfg.swizzle)
                T.clear(Cl)
                for k in T.Pipelined(k_tiles, num_stages=cfg.stages):
                    T.copy(A[by * BM, k * BK], As)
                    T.copy(W[bx * BN, k * BK], Ws)
                    T.gemm(As, Ws, Cl, transpose_B=True, policy=_POLICY)
                T.copy(Cl, Cs)
                for i, j in T.Parallel(BM, BN // 2):
                    gcol = (j // GU_BLOCK) * (2 * GU_BLOCK) + (j % GU_BLOCK)
                    g = T.cast(Cs[i, gcol], acc)
                    u = T.cast(Cs[i, gcol + GU_BLOCK], acc)
                    s = T.cast(g / (T.float32(1.0) + T.exp(-g)), dt)
                    Os[i, j] = T.cast(T.cast(s, acc) * u, dt)
                T.copy(Os, O[by * BM, bx * (BN // 2)])

    elif epilogue == "swiglu2":
        # Two weights, no packing, as tir/gemm/gate_up_T*.py: per K step one A tile and
        # two B tiles (BN rows of w_gate, BN rows of w_up at the same output columns)
        # arrive by TMA; two WGMMA chains fill acc_g and acc_u; the epilogue runs on the
        # accumulators in registers: O = bf16(bf16(silu(bf16(g))) * bf16(u)).
        HB = BN
        NO = N

        @T.prim_func
        def main(A: T.Tensor((M, K), dt), Wg: T.Tensor((N, K), dt), Wu: T.Tensor((N, K), dt), O: T.Tensor((M, NO), dt)):
            with T.Kernel(T.ceildiv(N, HB), T.ceildiv(M, BM), threads=cfg.threads) as (bx, by):
                As = T.alloc_shared((BM, BK), dt)
                Gs = T.alloc_shared((HB, BK), dt)
                Us = T.alloc_shared((HB, BK), dt)
                Cg = T.alloc_fragment((BM, HB), acc)
                Cu = T.alloc_fragment((BM, HB), acc)
                Ol = T.alloc_fragment((BM, HB), dt)
                Os = T.alloc_shared((BM, HB), dt)
                if cfg.swizzle:
                    T.use_swizzle(panel_size=cfg.swizzle)
                T.clear(Cg)
                T.clear(Cu)
                for k in T.Pipelined(k_tiles, num_stages=cfg.stages):
                    T.copy(A[by * BM, k * BK], As)
                    T.copy(Wg[bx * HB, k * BK], Gs)
                    T.copy(Wu[bx * HB, k * BK], Us)
                    T.gemm(As, Gs, Cg, transpose_B=True, policy=_POLICY)
                    T.gemm(As, Us, Cu, transpose_B=True, policy=_POLICY)
                for i, j in T.Parallel(BM, HB):
                    g = T.cast(T.cast(Cg[i, j], dt), acc)
                    u = T.cast(T.cast(Cu[i, j], dt), acc)
                    # rcp.approx for the division: bitwise equal to torch on the tests (WORKLOG)
                    s = T.cast(g * T.fast_rcp(T.float32(1.0) + T.exp(-g)), dt)
                    Ol[i, j] = T.cast(T.cast(s, acc) * u, dt)
                T.copy(Ol, Os)
                T.copy(Os, O[by * BM, bx * HB])

    elif epilogue == "splitk":
        S = cfg.split_k
        assert K % (S * BK) == 0
        KS = K // S
        ks_tiles = KS // BK

        @T.prim_func
        def main(A: T.Tensor((M, K), dt), W: T.Tensor((N, K), dt), P: T.Tensor((S, M, N), acc)):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), S, threads=cfg.threads) as (bx, by, bz):
                As = T.alloc_shared((BM, BK), dt)
                Ws = T.alloc_shared((BN, BK), dt)
                Cl = T.alloc_fragment((BM, BN), acc)
                if cfg.swizzle:
                    T.use_swizzle(panel_size=cfg.swizzle)
                T.clear(Cl)
                for k in T.Pipelined(ks_tiles, num_stages=cfg.stages):
                    T.copy(A[by * BM, bz * KS + k * BK], As)
                    T.copy(W[bx * BN, bz * KS + k * BK], Ws)
                    T.gemm(As, Ws, Cl, transpose_B=True, policy=_POLICY)
                T.copy(Cl, P[bz, by * BM, bx * BN])

    else:
        raise ValueError(epilogue)
    return _compile(main, f"gemm_{epilogue}_{M}x{N}x{K}_{cfg.tag()}")


@lru_cache(maxsize=None)
def splitk_reduce_kernel(M: int, N: int, S: int, residual: bool, block: int = 1024):
    """C = bf16(R + bf16(sum_s P[s]))  (or bf16(sum_s P[s]) without residual)."""
    dt, acc = "bfloat16", "float"
    total = M * N
    per_thread = 8
    span = block * per_thread

    if residual:

        @T.prim_func
        def main(P: T.Tensor((S, M, N), acc), R: T.Tensor((M, N), dt), C: T.Tensor((M, N), dt)):
            with T.Kernel(T.ceildiv(total, span), threads=block) as bx:
                for i in T.Parallel(span):
                    idx = bx * span + i
                    m = idx // N
                    n = idx % N
                    s = T.alloc_var(acc)
                    s = T.float32(0)
                    for z in T.serial(S):
                        s = s + P[z, m, n]
                    C[m, n] = T.cast(T.cast(R[m, n], acc) + T.cast(T.cast(s, dt), acc), dt)

    else:

        @T.prim_func
        def main(P: T.Tensor((S, M, N), acc), C: T.Tensor((M, N), dt)):
            with T.Kernel(T.ceildiv(total, span), threads=block) as bx:
                for i in T.Parallel(span):
                    idx = bx * span + i
                    m = idx // N
                    n = idx % N
                    s = T.alloc_var(acc)
                    s = T.float32(0)
                    for z in T.serial(S):
                        s = s + P[z, m, n]
                    C[m, n] = T.cast(s, dt)

    return _compile(main, f"splitk_reduce_{M}x{N}_s{S}{'_res' if residual else ''}")


# The GDN gate projections b and a (N = 48 each) as transformers computes them. A
# 1-ulp change in a moves the recurrent state through the decay g, so these two take
# the summation order of the library the reference runs: for N = 48, K = 5120 cuBLAS
# (CUDA 12, H200) launches nvjet_sm90_tst_16x64_64x16_*_splitK with grid z = 8 for
# M <= 512 and 4 for M = 1024 (torch.profiler trace) + cublasLt::splitKreduce_kernel.
# Reproduced as: contiguous K slices, one wgmma accumulator per 16-column tile, f32
# partials, summed from the last slice to the first (WORKLOG: b|a elements that differ
# from cuBLAS drop from 38 to 8 at T=128 layer 9; the packed GEMM's order is kept for
# qkv and z, which match cuBLAS bitwise).
BA_N = 96          # [w_b | w_a] rows


def ba_split(M: int) -> int:
    return 8 if M <= 512 else 4


def ba_bn(M: int) -> int:
    """Tile width of ba_splitk_kernel. The partials are bitwise the same for BN 16 / 32 /
    48 / 96 (the per-column K order does not depend on it); time us at T = 128 / 256 /
    512 / 1024: BN 16 4.83 / 5.28 / 9.00 / 14.69, BN 32 5.54 / 5.82 / 5.99 / 8.96,
    BN 48 6.06 / 6.13 / 6.20 / 9.34, BN 96 7.62 / 7.71 / 7.79 / 11.69."""
    return 16 if M <= 256 else 32


@lru_cache(maxsize=None)
def ba_splitk_kernel(M: int, K: int, S: int, BM: int = 128, BN: int = 16, BK: int = 64):
    """P[s, M, 96] = f32 partial of x @ [w_b | w_a]^T over the s-th contiguous K slice."""
    N = BA_N
    per = K // BK // S
    assert per * BK * S == K and M % BM == 0

    @T.prim_func
    def main(A: T.Tensor((M, K), "bfloat16"), B: T.Tensor((N, K), "bfloat16"), P: T.Tensor((S, M, N), "float32")):
        with T.Kernel(N // BN, M // BM, S, threads=256) as (bx, by, bz):
            As = T.alloc_shared((BM, BK), "bfloat16")
            Bs = T.alloc_shared((BN, BK), "bfloat16")
            C = T.alloc_fragment((BM, BN), "float32")
            T.clear(C)
            for k in T.Pipelined(per, num_stages=2):
                T.copy(A[by * BM, (bz * per + k) * BK], As)
                T.copy(B[bx * BN, (bz * per + k) * BK], Bs)
                T.gemm(As, Bs, C, transpose_B=True, policy=_POLICY)
            T.copy(C, P[bz, by * BM, bx * BN])

    return _compile(main, f"ba_splitk_{M}x{N}x{K}_s{S}")


@lru_cache(maxsize=None)
def ba_reduce_kernel(M: int, S: int, out_cols: int, col0: int):
    """Out[m, col0 + n] = bf16(((P[S-1] + P[S-2]) + ...) + P[0]) for n < 96: the b|a
    columns of the GDN projection buffer (row stride out_cols)."""
    N = BA_N

    @T.prim_func
    def main(P: T.Tensor((S, M, N), "float32"), Out: T.Tensor((M, out_cols), "bfloat16")):
        with T.Kernel(M, threads=N) as m:
            n = T.get_thread_binding()
            s = T.alloc_var("float32")
            s = P[S - 1, m, n]
            for z in T.serial(S - 1):
                s = T.ieee_add(s, P[S - 2 - z, m, n])
            Out[m, col0 + n] = T.cast(s, "bfloat16")

    return _compile(main, f"ba_reduce_{M}_s{S}")


def pack_gate_up(w_gate, w_up):
    """[I, H] + [I, H] -> [2I, H] with rows interleaved in blocks of GU_BLOCK."""
    import torch

    I, H = w_gate.shape
    g = w_gate.reshape(I // GU_BLOCK, GU_BLOCK, H)
    u = w_up.reshape(I // GU_BLOCK, GU_BLOCK, H)
    return torch.stack([g, u], dim=1).reshape(2 * I, H).contiguous()
