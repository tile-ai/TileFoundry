"""A DeepSeek-V3 style block-scaled FP8 GEMM as plain HIR.

A and B are e4m3; each 128-wide K block carries its own f32 scales: one per row
of A (1 x 128) and one per 128 x 128 tile of B. One CTA owns the whole
128 x 128 output. Every block's product is completed in f32 first, then scaled
by that block's row and tile scales and added into the f32 accumulator, so no
earlier block is scaled twice. B is the (K, N) view of a row-major (N, K)
weight, so its K windows are K-contiguous. The block offsets ``kb * BLOCK`` are
scalar index arithmetic, which costs nothing.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 128
N = 128
K = 512
BLOCK = 128
K_BLOCKS = K // BLOCK
N_BLOCKS = N // BLOCK


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 256)))
class FP8_BLOCK_SCALED_GEMM:
    @func
    def gemm(a: Tensor[(M, K), "fp8e4m3"],
             b: Tensor[(K, N), "fp8e4m3", Layout((K, N), (1, K))],
             a_scale: Tensor[(M, K_BLOCKS), "f32"],
             b_scale: Tensor[(K_BLOCKS, N_BLOCKS), "f32"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(1,), names=("g",)) as _cta:
            acc = tf.zeros(Tensor[(M, N), "f32", "rmem"])
            for kb in range(K_BLOCKS):
                at = tf.reshard(a[:, kb * BLOCK:kb * BLOCK + BLOCK], ((M, BLOCK), {}), "smem")
                bt = tf.reshard(b[kb * BLOCK:kb * BLOCK + BLOCK, :], ((BLOCK, N), {}), "smem")
                part = tf.matmul(at, bt, out_dtype="f32")
                row_scale = tf.reshard(a_scale[:, kb:kb + 1], ((M, 1), {}), "rmem")
                tile_scale = tf.reshard(b_scale[kb:kb + 1, :], ((1, N_BLOCKS), {}), "rmem")
                acc = acc + part * row_scale * tile_scale
            return tf.reshard(tf.cast(acc, "bf16"), ((M, N), {}), "gmem")
