"""The same outer M/N tiling, with every window written out instead of tiled.

``for m in tf.tile(M, BM)`` binds ``m`` to a window and ``a[m, k]`` is that window
read back.  An author who wants the offsets in their own hands writes the loop
as a ``range`` and the window as a slice --- ``a[m:m + BM, k]`` --- which names
the counter a second time, in a second ``Var`` equal to the loop's own, and
that mention is read as the counter it is: the emitted kernel is the tiled one,
plus the seed for the output. B is the (K, N) view of a row-major (N, K)
weight, so its windows have K at stride one and the atom states ``b_major=K``.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 128
N = 64
K = 32
BM = 64
BN = 32
BK = 16
STAGES = 2


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_EXPLICIT_WINDOWS:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(K, N), "bf16", Layout((K, N), (1, K))],
    ) -> Tensor[(M, N), "bf16"]:
        a_smem = Layout(((8, 8), (2, 8)), ((128, 8), (64, 1)))
        b_smem = Layout(((2, 8), (4, 8)), ((64, 1), (128, 8)))
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=32, dtype="bf16", form=T.cuda.sm90.Form.SS,
                    a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)

                out = tf.zeros(Tensor[(M, N), "bf16"])
                for m in range(0, M, BM):
                    for n in range(0, N, BN):
                        with Mesh(threads[1, :], layout=(4, 8, 4), names=('warp', 'lane8', 'lane4')) as _compute:
                            acc = tf.zeros(Tensor[(BM, BN), "f32", ((8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 4), (1, 8, 16, 64, 128, 512)), "rmem"])

                        for k in tf.tile(K, BK):
                            with threads[0, :32] as _loader:
                                lhs = tf.schedule(
                                    (a[m:m + BM, k],),
                                    op=T.copy_async_tensor(smem_layout=a_smem),
                                    buffers=STAGES,
                                )
                                rhs = tf.schedule(
                                    (b[k, n:n + BN],),
                                    op=T.copy_async_tensor(smem_layout=b_smem),
                                    buffers=STAGES,
                                )

                            with Mesh(threads[1, :], layout=(4, 8, 4), names=('warp', 'lane8', 'lane4')) as _compute:
                                acc = tf.schedule(
                                    (acc, lhs, rhs),
                                    op=T.tiled_mma(atom=wgmma),
                                )

                        with Mesh(threads[1, :], layout=(4, 8, 4), names=('warp', 'lane8', 'lane4')) as _compute:
                            tile_out = tf.cast(acc, dtype="bf16")
                            out = tf.insert_slice(out, tile_out, (m, n))
                return out
