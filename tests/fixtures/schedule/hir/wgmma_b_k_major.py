"""Score 64 query rows against 32 key rows with a K-major B descriptor.

Keys are stored row by row with the head dimension contiguous, which is how an
attention kernel holds them, so ``q @ k^T`` reads B with K at stride one. Binding
``b_major`` selects that descriptor: the 128-byte-swizzled key tile is staged as
it lies in memory, each k16 issue reads its 16 contiguous dimensions at a
swizzle offset, and the 32 key rows step by one swizzle row.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split, Swizzle
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 64
SW128 = Swizzle(3, 4, 3)

Q_SMEM = ComposedLayout(SW128, 0, Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))))
K_SMEM = ComposedLayout(SW128, 0, Layout(((4, 16), (4, 8)), ((16, 1), (512, 64))))
KEYS_T = Layout((K, N), (1, K))
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
class WGMMA_B_K_MAJOR:
    @func
    def gemm(
        q: Tensor[(M, K), "bf16"],
        keys_t: Tensor[(K, N), "bf16", KEYS_T],
    ) -> Tensor[(M, N), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=N,
                    form=T.cuda.sm90.Form.SS,
                    a_major=T.cuda.sm90.Major.K,
                    b_major=T.cuda.sm90.Major.K,
                )

                with threads[1, :] as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ACC, "rmem"])

                with threads[0, :32] as _loader:
                    lhs = tf.schedule((q[:, 0:K],), op=T.copy_async(smem_layout=Q_SMEM))
                    rhs = tf.schedule((keys_t[:, 0:N],), op=T.copy_async(smem_layout=K_SMEM))

                with threads[1, :] as _compute:
                    acc = tf.schedule(
                        (acc, lhs, rhs),
                        op=T.tiled_mma(atom=wgmma),
                        repeat=(1, 1, 4),
                    )
                    result = tf.cast(acc, dtype="bf16")
                return result
