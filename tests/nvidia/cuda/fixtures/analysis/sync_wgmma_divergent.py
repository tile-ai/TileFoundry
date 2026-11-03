"""A chained WGMMA inside a loop whose trip count depends on the issuing thread.

The WGMMA before the loop takes the fence, and the one in the loop chains on its
accumulator, so the loop holds no step. In ``WgmmaDivergent`` the trip count is
the warp, so the warpgroup's threads would not all issue the WGMMA together, and
the sync family refuses the Function. In ``WgmmaGroupLoop`` it is the warpgroup,
so two warpgroups loop a different number of times, each of them as a whole.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

A = ComposedLayout(Swizzle(3, 4, 3), 0, Layout(((8, 8), 16), ((512, 64), 1)))
B = ComposedLayout(
    Swizzle(3, 4, 3), 0, Layout(((2, 8), (2, 1, 64)), ((512, 64), (4096, 1024, 1)))
)


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 128)),
)
class WgmmaDivergent:
    @func
    def gemm(
        a: Tensor[(64, 16), "bf16"], b: Tensor[(16, 16), "bf16"]
    ) -> Tensor[(64, 16), "f32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as threads:
                output = tf.zeros(Tensor[(64, 16), "f32"])
                acc = tf.zeros(Tensor[
                    (64, 16), "f32",
                    ((4 @ threads.warp, 2, 8 @ threads.lane8, 2, 4 @ threads.lane4, 2,),
                     (16, 8, 1, 512, 128, 64)), "rmem",
                ])
                lhs = tf.schedule(
                    (a, tf.empty(Tensor[
                        (64, 16), "bf16", Layout(((8, 8), (2, 8)), ((128, 8), (64, 1))), "smem",
                    ])),
                    op=T.copy(),
                )
                rhs = tf.schedule(
                    (tf.transpose(b, (1, 0)), tf.empty(Tensor[
                        (16, 16), "bf16", Layout(((2, 8), (2, 8)), ((64, 1), (128, 8))), "smem",
                    ])),
                    op=T.copy(),
                )
                wgmma = T.cuda.sm90.Wgmma(
                    n=16, dtype="bf16", form=T.cuda.sm90.Form.SS,
                    a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K,
                )
                acc = tf.schedule((acc, lhs, rhs), op=T.tiled_mma(atom=wgmma))
                for _j in range(threads.warp):
                    acc = tf.schedule((acc, lhs, rhs), op=T.tiled_mma(atom=wgmma))
                return tf.insert_slice(output, acc, (0, 0))


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 256)),
)
class WgmmaGroupLoop:
    @func
    def gemm(
        a: Tensor[(64, 16), "bf16"], b: Tensor[(16, 128), "bf16"]
    ) -> Tensor[(64, 128), "f32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(
                ("thread",), layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")
            ) as threads:
                output = tf.zeros(Tensor[(64, 128), "f32"])
                acc = tf.zeros(Tensor[
                    (64, 128), "f32",
                    ((4 @ threads.warp, 2, 8 @ threads.lane8, 2 @ threads.group, 8,
                      4 @ threads.lane4, 2),
                     (16, 8, 1, 4096, 512, 128, 64)), "rmem",
                ])
                lhs = tf.schedule((a, tf.empty(Tensor[(64, 16), "bf16", A, "smem"])), op=T.copy())
                rhs = tf.schedule((b, tf.empty(Tensor[(16, 128), "bf16", B, "smem"])), op=T.copy())
                wgmma = T.cuda.sm90.Wgmma(
                    n=64, dtype="bf16", form=T.cuda.sm90.Form.SS,
                    a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.MN,
                )
                acc = tf.schedule((acc, lhs, rhs), op=T.tiled_mma(atom=wgmma))
                for _j in range(threads.group + 1):
                    acc = tf.schedule((acc, lhs, rhs), op=T.tiled_mma(atom=wgmma))
                return tf.insert_slice(output, acc, (0, 0))
