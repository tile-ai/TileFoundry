"""Scheduled HIR: gdn_in at prompt length 256.

Implements: Qwen3_8DeltaNetLayer.gated_delta_net: matmul(x, w_qkv|w_z|w_b|w_a, NK) (packed rows, padded to 16640)
Plan row: runtime/plan.py, length 256: GemmConfig(BM=128, BN=256, BK=64, stages=4, threads=256, swizzle=8, split_k=1)
Finalized TIR: tir/gemm/gdn_in_T256.py   Runtime kernel: runtime/kernels/gemm.py gemm_kernel(..., "store")

Structure: one CTA per (128 x 256) output tile, tiles launched M-fastest;
384 threads = warpgroup 0
(lanes 0..31 issue TMA) + warpgroups 1..2 (WGMMA m64n256k16, warpgroup g owns rows
[64g, 64g+64)); A and W tiles stream through 4-deep 128B-swizzled TMA rings; W is
the checkpoint's (out, in) weight, read K-major (b_major=K) through a bitcast view.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 256
N = 16640
K = 5120
BM = 128
BN = 256
BK = 64
STAGES = 4
GM = M // BM
GN = N // BN


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", GM * GN), Topology("thread", 384)))
class SCHED_GDN_IN_T256:
    @func
    def gemm(a: Tensor[(M, K), "bf16"], w: Tensor[(N, K), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(GN, GM), names=("bn", "bm")) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                          a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                out = tf.zeros(Tensor[(M, N), "bf16"])
                m = cta.bm * BM
                n = cta.bn * BN
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 32), (16384, 1, 8, 16, 64, 128, 512)), "rmem"])
                for k in tf.tile(0, K, BK):
                    with threads[0, :32] as _loader:
                        lhs_ring = tf.empty(Tensor[(BM, BK), "bf16", Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        lhs = tf.schedule((a[m:m + BM, k], lhs_ring), op=T.copy_async_tensor())
                        rhs_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        rhs_0 = tf.schedule((w[n:n + BN, k], rhs_ring), op=T.copy_async_tensor())
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        rhs_t0 = tf.bitcast(rhs_0, Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))
                        acc = tf.schedule((acc, lhs, rhs_t0), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    tile_out = tf.cast(acc, dtype="bf16")
                    staged_dst = tf.empty(Tensor[(BM, 256), "bf16", (BM, 256), "smem"])
                    staged = tf.schedule((tile_out, staged_dst), op=T.copy())
                with threads[0, :32] as _storer:
                    tile_dst = tf.empty(Tensor[(BM, 256), "bf16", (BM, 256)])
                    tile = tf.schedule((staged, tile_dst), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (m, n))
                return out
