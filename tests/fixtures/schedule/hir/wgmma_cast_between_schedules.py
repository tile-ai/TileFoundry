"""Author an elementwise and reduction epilogue between scheduled instructions.

Each K tile of ``b_f32`` is copied into registers held by the loader warp,
narrowed there, and copied into shared memory. The author therefore owns the
three storage transitions; lowering does not invent a distribution for an
unscheduled whole-tensor cast. The accumulator epilogue then exercises
automatically selected binary, ReLU, reduction, and cast instructions alongside
an explicit reduction, producing an ``(M, 1)`` result.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

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
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 128),
                names=("role", "participant"),
            ) as threads:
                wgmma = T.cuda.sm90.Wgmma(
                    n=32, dtype="bf16", form=T.cuda.sm90.Form.SS, a_major=T.cuda.sm90.Major.K)

                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                    acc = tf.zeros(Tensor[(M, N), "f32", ((4 @ _compute.warp, 2, 8 @ _compute.lane8, 4, 4 @ _compute.lane4, 2,), (16, 8, 1, 512, 128, 64)), "rmem"])

                for k in tf.tile(K, BK):
                    with Mesh(threads[0, :32], layout=(32,), names=("lane",)) as _loader:
                        lhs_ring = tf.empty(
                            Tensor[
                                (64, 16), "bf16",
                                Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))),
                                "smem",
                            ],
                            buffers=STAGES,
                        )
                        lhs = tf.schedule(
                            (a[:, k], lhs_ring),
                            op=T.copy_async_tensor(),
                        )

                    with Mesh(threads[0, :32], layout=(32,), names=("lane",)) as _loader:
                        b_tile_dst = tf.empty(
                            Tensor[
                                (16, 32), "f32",
                                ShardLayout(
                                    layout=Layout((16, 32), (1, 16)),
                                    attrs=(S(1),),
                                    mesh=_loader,
                                ),
                                "rmem",
                            ],
                        )
                        b_tile = tf.schedule(
                            (b_f32[k, :], b_tile_dst),
                            op=T.copy(),
                        )
                        b = tf.cast(b_tile, dtype="bf16")
                        rhs_ring = tf.empty(
                            Tensor[
                                (16, 32), "bf16",
                                Layout(((2, 8), (4, 8)), ((64, 8), (128, 1))),
                                "smem",
                            ],
                            buffers=STAGES,
                        )
                        rhs = tf.schedule(
                            (b, rhs_ring),
                            op=T.copy(),
                        )

                    with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                        acc = tf.schedule(
                            (acc, lhs, rhs),
                            op=T.tiled_mma(atom=wgmma),
                        )

                output = tf.zeros(Tensor[(M, 1), "bf16"])
                with Mesh(threads[1, :], layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _compute:
                    bias_r_dst = tf.empty(
                        Tensor[
                            (64, 32), "f32",
                            ShardLayout(
                                layout=Layout((4, 2, 8, 4, 4, 2,), (16, 8, 1, 512, 128, 64)),
                                attrs=(S(0), S(2), S(4)),
                                mesh=_compute,
                            ),
                            "rmem",
                        ],
                    )
                    bias_r = tf.schedule((bias, bias_r_dst), op=T.copy())
                    acc = tf.relu(acc + bias_r)
                    explicit_dst = tf.empty(
                        Tensor[
                            (64, 1), "f32",
                            ShardLayout(
                                layout=Layout((8, 2, 4, 1, 1, 1), (8, 4, 1, 0, 0, 0)),
                                attrs=(S(2), S(0), B()),
                                mesh=_compute,
                            ),
                            "rmem",
                        ],
                    )
                    explicit = tf.schedule(
                        (acc, explicit_dst),
                        op=T.reduce(
                            axes=(1,), keepdim=True, kind=ReduceKind.SUM
                        ),
                    )
                    automatic = tf.reduce(
                        acc, axes=(1,), keepdim=True, kind=ReduceKind.SUM
                    )
                    result = tf.cast(explicit + automatic, dtype="bf16")
                    result = tf.insert_slice(output, result, (0, 0))
                return result
