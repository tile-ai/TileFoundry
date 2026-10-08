"""A BK=64 authored schedule: one WGMMA repeats over K as well as M.

The staged tile is four atoms deep, so each transfer moves 16 KB instead of
4 KB. A uses one K-major 128-byte row whose four 16-element slices begin at
offsets 0, 16, 32, and 48; B is tiled four times over K. The schedule issues
two M groups and four K slices into one accumulator before the next stage.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split, Swizzle
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 128
N = 16
K = 128
BK = 64
STAGES = 2


_COMPUTE = ThreadMesh((Topology("thread", 384),),
                      ComposedLayout(None, 128, Layout((2, 4, 8, 4), (128, 32, 4, 1))),
                      ("group", "warp", "lane8", "lane4"))
A_SMEM = ComposedLayout(Swizzle(3, 4, 3), 0,
                        Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))))
B_SMEM = Layout(((4, 2, 8), (2, 8)), ((256, 64, 8), (128, 1)))
ACC = ShardLayout(Layout((2, 8, 2, 4, 2, 4, 2),
                         (1024, 1, 8, 16, 64, 128, 512)),
                  (Split(0), Split(3), Split(1), Split(5)), _COMPUTE)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 384)),
)
class WGMMA_REPEAT_ALONG_K:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(3, 128),
                names=("warpgroup", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=16, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                with threads[1:3, :] as _compute:
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

                    with threads[1:3, :] as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                            repeat=(2, 1, 4),
                        )

                with threads[1:3, :] as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
