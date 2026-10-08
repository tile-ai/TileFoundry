"""Run a two-stage 64x32 WGMMA with a K-major A descriptor.

Binding ``a_major`` selects a descriptor whose core matrix runs along K, so
the staged A tile and its source both have K at stride one. In each k iteration,
the loader copies only that iteration's (BK, M) input window into a compact
(M, BK) gmem tile before TMA. The 64x16 fragment
uses two core-matrix offsets while B remains MN-major and the accumulator keeps
the standard 64x32 register arrangement.
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

A_SMEM = Layout(((8, 8), (2, 8)), ((128, 8), (64, 1)))
B_SMEM = Layout(((2, 8), (4, 8)), ((64, 8), (128, 1)))
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
class WGMMA_A_K_MAJOR:
    @func
    def gemm(
        at: Tensor[(K, M), "bf16"],
        b: Tensor[(K, N), "bf16"],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                with threads[1, :] as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ACC, "rmem"])

                for k in tile(K, BK):
                    with threads[0, :32] as _loader:
                        a = tf.transpose(at[k, :], (1, 0))
                        lhs = tf.schedule(
                            (a,),
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
