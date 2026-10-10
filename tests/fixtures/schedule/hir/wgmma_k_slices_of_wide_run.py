"""A distinct authored schedule: 2-stage TMA + one 64x32 WGMMA group.

A is read K-major in the 64-byte mode, which is wider than one atom's K: a
staged tile's K is 32 elements, one swizzled run of them per row, and each
64x16 atom reads its slice of that run -- ``[0, 16)`` and then ``[16, 32)``,
the descriptor starting that many elements further in.  So the two atoms
of one stage are two statements, one per slice, and the view the second
reads starts 16 elements into the run.  The atom is the same one both times:
where a slice starts is the view's, and the descriptor reads it there.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 64
BK = 32
STAGES = 2


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_K_SLICES_OF_WIDE_RUN:
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
                    n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                with Mesh(threads[1, :], layout=(4, 8, 4), names=('warp', 'lane8', 'lane4')) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 4), (1, 8, 16, 64, 128, 512)), "rmem"])

                for k in tf.tile(K, BK):
                    with threads[0, :32] as _loader:
                        lhs = tf.schedule(
                            (a[:, k],),
                            op=T.copy_async_tensor(smem_layout=Layout(((8, 8), (2, 16)), ((256, 32), (16, 1))) | Swizzle(2, 4, 3)),
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b[k, :],),
                            op=T.copy_async_tensor(smem_layout=Layout(((2, 2, 8), (4, 8)), ((512, 64, 8), (128, 1)))),
                            buffers=STAGES,
                        )

                    with Mesh(threads[1, :], layout=(4, 8, 4), names=('warp', 'lane8', 'lane4')) as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                        )

                with Mesh(threads[1, :], layout=(4, 8, 4), names=('warp', 'lane8', 'lane4')) as _compute:
                    result = tf.cast(acc, dtype="bf16")
                return result
