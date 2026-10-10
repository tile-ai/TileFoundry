"""Author an elementwise and reduction epilogue between scheduled instructions.

Each K tile of ``b_f32`` is copied into registers held by the loader warp,
narrowed there, and copied into shared memory. The author therefore owns the
three storage transitions; lowering does not invent a distribution for an
unscheduled whole-tensor cast. The accumulator epilogue then exercises
automatically selected binary, ReLU, reduction, and cast instructions alongside
an explicit reduction, producing an ``(M, 1)`` result.
"""

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 64
N = 32
K = 32
BK = 16
STAGES = 2


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WGMMA_CAST_BETWEEN_SCHEDULES:
    @func
    def gemm(
        a: Tensor[(M, K), "bf16"],
        b_f32: Tensor[(K, N), "f32"],
        bias: Tensor[(M, N), "f32"],
    ) -> Tensor[(M, 1), "bf16", "umat"]:
        a_smem = Layout(((8, 8), (2, 8)), ((128, 8), (64, 1)))
        b_smem = Layout(((2, 8), (4, 8)), ((64, 8), (128, 1)))
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
                    with Mesh(threads[0, :32], layout=(32,), names=('lane',)) as _loader:
                        lhs = tf.schedule(
                            (a[:, k],),
                            op=T.copy_async_tensor(smem_layout=a_smem),
                            buffers=STAGES,
                        )

                    with Mesh(threads[0, :32], layout=(32,), names=('lane',)) as _loader:
                        b_tile = tf.schedule(
                            (b_f32[k, :],),
                            op=T.copy(rmem_layout=((32 @ _loader.lane, 16), (16, 1))),
                        )
                        b = tf.cast(b_tile, dtype="bf16")
                        rhs = tf.schedule(
                            (b,),
                            op=T.copy(smem_layout=b_smem),
                            buffers=STAGES,
                        )

                    with Mesh(threads[1, :], layout=(4, 8, 4), names=('warp', 'lane8', 'lane4')) as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                        )

                with Mesh(threads[1, :], layout=(4, 8, 4), names=('warp', 'lane8', 'lane4')) as _compute:
                    bias_r = tf.schedule((bias,), op=T.copy(rmem_layout=((8 @ _compute.lane8, 2, 4 @ _compute.warp, 2, 4 @ _compute.lane4, 4), (1, 8, 16, 64, 128, 512))))
                    acc = tf.relu(acc + bias_r)
                    explicit = tf.schedule(
                        (acc,),
                        op=T.reduce(
                            axes=(1,), keepdim=True, kind=ReduceKind.SUM
                        ),
                    )
                    automatic = tf.reduce(
                        acc, axes=(1,), keepdim=True, kind=ReduceKind.SUM
                    )
                    result = tf.cast(explicit + automatic, dtype="bf16")
                return result
