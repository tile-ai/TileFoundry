"""Schedule the block-scaled FP8 GEMM: TMA loads, FP8 WGMMA, per-block scaling.

Each 128-wide K block is staged by TMA: A as a 128 x 128 K-major tile with
128-byte swizzle, B from the K-contiguous view of the (N, K) weight. Two
warpgroups own 64 rows each; one FP8 WGMMA issue reads 32 bytes of K, so a
block is two M groups by four K slices into a fresh f32 block product. Only
then are the block's row and tile scales copied into registers, multiplied in,
and the scaled block added to the f32 accumulator.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 128
N = 128
K = 512
BLOCK = 128
K_BLOCKS = K // BLOCK
N_BLOCKS = N // BLOCK
STAGES = 2


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 384)),
)
class FP8_BLOCK_SCALED_GEMM:
    @func
    def gemm(
        a: Tensor[(M, K), "fp8e4m3"],
        b: Tensor[(K, N), "fp8e4m3", Layout((K, N), (1, K))],
        a_scale: Tensor[(M, K_BLOCKS), "f32"],
        b_scale: Tensor[(K_BLOCKS, N_BLOCKS), "f32"],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        a_smem = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))))
        b_smem = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))))
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS)

                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 16), (8192, 1, 8, 16, 64, 128, 512)), "rmem"])

                for kb in range(K_BLOCKS):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, kb * BLOCK:kb * BLOCK + BLOCK],),
                            op=T.copy_async_tensor(smem_layout=a_smem),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[kb * BLOCK:kb * BLOCK + BLOCK, :],),
                            op=T.copy_async_tensor(smem_layout=b_smem),
                            buffers=STAGES,
                        )

                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                        part = tf.zeros(Tensor[(M, N), "f32", ((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 16), (8192, 1, 8, 16, 64, 128, 512)), "rmem"])
                        part = tf.schedule(
                            (part, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                            repeat=(2, 1, 4),
                        )
                        row_scale = tf.schedule(
                            (a_scale[:, kb:kb + 1],), op=T.copy(rmem_layout=((2 @ _compute.group, 8 @ _compute.lane8, 2, 4 @ _compute.warp), (64, 1, 8, 16)))
                        )
                        tile_scale = tf.schedule(
                            (b_scale[kb:kb + 1, :],), op=T.copy(rmem_layout=((1, 1), (1, 1), {_compute.group @ B()}))
                        )
                        acc = acc + part * row_scale * tile_scale

                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=('group', 'warp', 'lane8', 'lane4')) as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
