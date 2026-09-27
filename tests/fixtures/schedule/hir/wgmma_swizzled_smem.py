"""Stage a two-stage 64x32 WGMMA through swizzled shared memory.

The transfer layout states each descriptor's contiguous run. B spans 32 bf16
values across N and therefore uses its widest 64-byte swizzle; A remains the
unswizzled K-major descriptor, whose 16-element K row is 32 bytes. The atom and
accumulator otherwise match the K-major baseline fixture.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split, Swizzle
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 32
BK = 16
STAGES = 2

A_SMEM = Layout(((8, 8), (2, 8)), ((128, 8), (64, 1)))
B_SMEM = ComposedLayout(Swizzle(2, 4, 3), 0,
                        Layout(((2, 8), (1, 32)), ((256, 32), (512, 1))))
ACC = ShardLayout(
    Layout((8, 2, 4, 2, 4, 4), (1, 8, 16, 64, 128, 512)),
    (Split(2), Split(0), Split(4)),
    ThreadMesh((Topology("thread", 256),),
               ComposedLayout(None, 128, Layout((4, 8, 4), (32, 4, 1))),
               ("warp", "lane8", "lane4")),
)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_SWIZZLED_SMEM:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=32, form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

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
                    result = tf.cast(acc, dtype="bf16")
                return result
