"""One tile written into a larger output: the seeded zeros is not dead.

The counterpart to ``wgmma_insert_tiles_into_output``. There the M/N loops tile the whole
result, so the ``tf.zeros`` the author seeded it with is overwritten to the last
element and the memset it asks for is waste. Here there is one tile and no outer
loop, so every element of the output outside that 64x32 window is what the zeros
says it is -- and if the zeros is dropped, that part of the output is whatever
the caller happened to leave in it, which is a different program from the one
that was written.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 32
BK = 16
STAGES = 2
OUT_M = 128
OUT_N = 64


_COMPUTE = ThreadMesh((Topology("thread", 256),),
                      ComposedLayout(None, 128, Layout((4, 8, 4), (32, 4, 1))),
                      ("warp", "lane8", "lane4"))
A_SMEM = Layout(((8, 8), (2, 8)), ((128, 8), (64, 1)))
B_SMEM = Layout(((2, 8), (4, 8)), ((64, 8), (128, 1)))
ACC = ShardLayout(Layout((8, 2, 4, 2, 4, 4),
                         (1, 8, 16, 64, 128, 512)),
                  (Split(2), Split(0), Split(4)), _COMPUTE)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_ONE_TILE_OF_LARGER_OUTPUT:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(OUT_M, OUT_N), "bf16"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=32, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                out = tf.zeros(Tensor[(OUT_M, OUT_N), "bf16"])

                with threads[1, :] as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ACC, "rmem"])

                for k in tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, k],),
                            op=T.copy_async_tensor(smem_layout=A_SMEM),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :],),
                            op=T.copy_async_tensor(smem_layout=B_SMEM),
                            buffers=STAGES,
                        )

                    with threads[1, :] as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                        )

                with threads[1, :] as _compute:
                    tile_out = tf.cast(acc, dtype="bf16")
                    result = tf.insert_slice(out, tile_out, (0, 0))
                return result
