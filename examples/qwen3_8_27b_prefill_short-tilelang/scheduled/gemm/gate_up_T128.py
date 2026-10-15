"""Scheduled HIR: gate_up at prompt length 128.

Implements: Qwen3_8*Layer.swiglu: silu(matmul(x, w_gate, NK)) * matmul(x, w_up, NK) N is 17408; the program is stated over 17424 = 121 whole tiles: weight rows >= 17408 are TMA out-of-bounds zero fill and output columns >= 17408 are clipped by the TMA store, so the valid region is the authored computation.
Plan row: runtime/plan.py, length 128: GemmConfig(BM=128, BN=144, BK=128, stages=2, threads=256, swizzle=0, split_k=1)
Finalized TIR: tir/gemm/gate_up_T128.py   Runtime kernel: runtime/kernels/gemm.py gemm_kernel(..., "swiglu2")

Structure: one CTA per (128 x 144) output tile, tiles launched M-fastest;
384 threads = warpgroup 0
(lanes 0..31 issue TMA) + warpgroups 1..2 (WGMMA m64n144k16, warpgroup g owns rows
[64g, 64g+64)); A and W tiles stream through 2-deep 128B-swizzled TMA rings; W is
the checkpoint's (out, in) weight, read K-major (b_major=K) through a bitcast view.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 128
N = 17424
K = 5120
BM = 128
BN = 144
BK = 128
STAGES = 2
GM = M // BM
GN = N // BN


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", GM * GN), Topology("thread", 384)))
class SCHED_GATE_UP_T128:
    @func
    def gemm(a: Tensor[(M, K), "bf16"], w_gate: Tensor[(N, K), "bf16"], w_up: Tensor[(N, K), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(GM, GN), names=("bm", "bn")) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                          a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                out = tf.zeros(Tensor[(M, N), "bf16"])
                m = cta.bm * BM
                n = cta.bn * BN
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    acc_g = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 18), (9216, 1, 8, 16, 64, 128, 512)), "rmem"])
                    acc_u = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 18), (9216, 1, 8, 16, 64, 128, 512)), "rmem"])
                for k in tf.tile(0, K, BK):
                    with threads[0, :32] as _loader:
                        lhs_ring = tf.empty(Tensor[(BM, BK), "bf16", Layout(((2, 8, 8), (2, 4, 16)), ((4096, 512, 64), (8192, 16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        lhs = tf.schedule((a[m:m + BM, k], lhs_ring), op=T.copy_async_tensor())
                        gate_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        rhs_0 = tf.schedule((w_gate[n:n + BN, k], gate_ring), op=T.copy_async_tensor())
                        up_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((18, 8), (2, 4, 16)), ((512, 64), (9216, 16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        rhs_1 = tf.schedule((w_up[n:n + BN, k], up_ring), op=T.copy_async_tensor())
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        rhs_t0 = tf.bitcast(rhs_0, Layout(((2, 4, 16), (18, 8)), ((9216, 16, 1), (512, 64))) | Swizzle(3, 4, 3))
                        acc_g = tf.schedule((acc_g, lhs, rhs_t0), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 8))
                        rhs_t1 = tf.bitcast(rhs_1, Layout(((2, 4, 16), (18, 8)), ((9216, 16, 1), (512, 64))) | Swizzle(3, 4, 3))
                        acc_u = tf.schedule((acc_u, lhs, rhs_t1), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 8))
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    g = tf.cast(tf.cast(acc_g, dtype="bf16"), dtype="f32")
                    u = tf.cast(tf.cast(acc_u, dtype="bf16"), dtype="f32")
                    act = tf.cast(g / (tf.exp(tf.neg(g)) + 1.0), dtype="bf16")
                    tile_out = tf.cast(tf.cast(act, dtype="f32") * u, dtype="bf16")
                    staged_dst = tf.empty(Tensor[(BM, 144), "bf16", (BM, 144), "smem"])
                    staged = tf.schedule((tile_out, staged_dst), op=T.copy())
                with threads[0, :32] as _storer:
                    tile_dst = tf.empty(Tensor[(BM, 144), "bf16", (BM, 144)])
                    tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (m, n))
                return out
