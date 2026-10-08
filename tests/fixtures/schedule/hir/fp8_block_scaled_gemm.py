"""Schedule the block-scaled FP8 GEMM: TMA loads, FP8 WGMMA, per-block scaling.

Each 128-wide K block is staged by TMA: A as a 128 x 128 K-major tile with
128-byte swizzle, B from the K-contiguous view of the (N, K) weight. Two
warpgroups own 64 rows each; one FP8 WGMMA issue reads 32 bytes of K, so a
block is two M groups by four K slices into a fresh f32 block product. Only
then are the block's row and tile scales copied into registers, multiplied in,
and the scaled block added to the f32 accumulator.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.ir.types import (
    Broadcast,
    ComposedLayout,
    Layout,
    ShardLayout,
    Split,
    Swizzle,
)
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 128
N = 128
K = 512
BLOCK = 128
K_BLOCKS = K // BLOCK
N_BLOCKS = N // BLOCK
STAGES = 2
WEIGHT_VIEW = Layout((K, N), (1, K))

_COMPUTE = ThreadMesh((Topology("thread", 384),),
                      ComposedLayout(None, 128, Layout((2, 4, 8, 4), (128, 32, 4, 1))),
                      ("group", "warp", "lane8", "lane4"))
A_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0,
                        Layout(((2, 8, 8), (4, 32)), ((8192, 1024, 128), (32, 1))))
B_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0,
                        Layout(((4, 32), (16, 8)), ((32, 1), (1024, 128))))
ACC = ShardLayout(Layout((2, 8, 2, 4, 2, 4, 16),
                         (8192, 1, 8, 16, 64, 128, 512)),
                  (Split(0), Split(3), Split(1), Split(5)), _COMPUTE)
ROW = ShardLayout(Layout((2, 8, 2, 4), (64, 1, 8, 16)),
                  (Split(0), Split(3), Split(1), Broadcast()), _COMPUTE)
TILE = ShardLayout(Layout((1, 1), (1, 1)),
                   (Broadcast(), Broadcast(), Broadcast(), Broadcast()), _COMPUTE)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 384)),
)
class FP8_BLOCK_SCALED_GEMM:
    @func
    def gemm(
        a: Tensor[(M, K), "fp8e4m3"],
        b: Tensor[(K, N), "fp8e4m3", WEIGHT_VIEW],
        a_scale: Tensor[(M, K_BLOCKS), "f32"],
        b_scale: Tensor[(K_BLOCKS, N_BLOCKS), "f32"],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=128, dtype="fp8e4m3", form=T.cuda.sm90.Form.SS)

                with threads[1:3, :] as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ACC, "rmem"])

                for kb in range(K_BLOCKS):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, kb * BLOCK:kb * BLOCK + BLOCK],),
                            op=T.copy_async_tensor(smem_layout=A_SMEM),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[kb * BLOCK:kb * BLOCK + BLOCK, :],),
                            op=T.copy_async_tensor(smem_layout=B_SMEM),
                            buffers=STAGES,
                        )

                    with threads[1:3, :] as _compute:
                        part = tf.zeros(Tensor[(M, N), "f32", ACC, "rmem"])
                        part = tf.schedule(
                            (part, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                            repeat=(2, 1, 4),
                        )
                        row_scale = tf.schedule(
                            (a_scale[:, kb:kb + 1],), op=T.copy(rmem_layout=ROW)
                        )
                        tile_scale = tf.schedule(
                            (b_scale[kb:kb + 1, :],), op=T.copy(rmem_layout=TILE)
                        )
                        acc = acc + part * row_scale * tile_scale

                with threads[1:3, :] as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
