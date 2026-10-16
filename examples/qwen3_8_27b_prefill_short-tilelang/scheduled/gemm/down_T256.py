"""Scheduled HIR: down at prompt length 256.

Implements: Qwen3_8*Layer.layer: attended + matmul(silu(gate)*up, w_down, NK)
Plan row: runtime/plan.py, length 256: GemmConfig(BM=128, BN=160, BK=64, stages=4, threads=256, swizzle=8, split_k=2)
Finalized TIR: tir/gemm/down_T256.py   Runtime kernel: runtime/kernels/gemm.py gemm_kernel(..., "splitk")

Structure: one CTA per (128 x 160) output tile, K split in 2 slices (f32 partial sums, then a reduce + residual pass), tiles launched M-fastest;
384 threads = warpgroup 0
(lanes 0..31 issue TMA) + warpgroups 1..2 (WGMMA m64n160k16, warpgroup g owns rows
[64g, 64g+64)); A and W tiles stream through 4-deep 128B-swizzled TMA rings; W is
the checkpoint's (out, in) weight, read K-major (b_major=K) through a bitcast view.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 256
N = 5120
K = 17408
BM = 128
BN = 160
BK = 64
STAGES = 4
GM = M // BM
GN = N // BN
SPLIT = 2
KS = K // SPLIT


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", GM * GN * SPLIT), Topology("thread", 384)))
class SCHED_DOWN_T256:
    @func
    def gemm(a: Tensor[(M, K), "bf16"], w: Tensor[(N, K), "bf16"]) -> Tensor[(SPLIT, M, N), "f32"]:
        with Mesh(("cta",), layout=(SPLIT, GN, GM), names=("ks", "bn", "bm")) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                          a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                out = tf.zeros(Tensor[(SPLIT, M, N), "f32"])
                m = cta.bm * BM
                n = cta.bn * BN
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 20, 4 @ _compute.lane4, 2), (10240, 16, 8, 1, 512, 128, 64)), "rmem"])
                for kk in range(0, KS, BK):
                    with threads[0, :32] as _loader:
                        lhs_ring = tf.empty(Tensor[(BM, BK), "bf16", Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        lhs = tf.schedule((a[m:m + BM, cta.ks * KS + kk:cta.ks * KS + kk + BK], lhs_ring), op=T.copy_async_tensor())
                        rhs_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        rhs_0 = tf.schedule((w[n:n + BN, cta.ks * KS + kk:cta.ks * KS + kk + BK], rhs_ring), op=T.copy_async_tensor())
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        rhs_t0 = tf.bitcast(rhs_0, Layout(((4, 16), (20, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))
                        acc = tf.schedule((acc, lhs, rhs_t0), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    tile_out = acc
                    tile_dst = tf.empty(Tensor[(1, BM, BN), "f32", (1, BM, BN)])
                    tile = tf.schedule((tf.reshape(tile_out, new_shape=(1, BM, BN)), tile_dst), op=T.copy())
                    out = tf.insert_slice(out, tile, (cta.ks, m, n))
                return out
