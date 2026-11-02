"""A retained callee reads a WGMMA accumulator nothing has waited for yet.

The caller issues one SS WGMMA into ``acc`` and passes the result straight to
``drain``, which copies the accumulator registers out. The call is the first
access after the WGMMA, so the wait for the warpgroup's group comes before it.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

WARPGROUP = Mesh(
    (Topology("thread", 128),), Layout((4, 8, 4), (32, 4, 1)), ("warp", "lane8", "lane4")
)
ACC = ShardLayout(
    layout=Layout((4, 2, 8, 2, 4, 2), (16, 8, 1, 512, 128, 64)),
    attrs=(S(0), S(2), S(4)),
    mesh=WARPGROUP,
)


@module(entry="drain")
class Drain:
    @func
    def drain(acc: Tensor[(64, 16), "f32", ACC, "rmem"]) -> Tensor[(64, 16), "f32", "rmem"]:
        with Mesh(
            (Topology("thread", 128),), layout=(4, 8, 4), names=("warp", "lane8", "lane4")
        ) as threads:
            held = tf.empty(Tensor[
                (64, 16), "f32",
                ((4 @ threads.warp, 2, 8 @ threads.lane8, 2, 4 @ threads.lane4, 2,),
                 (16, 8, 1, 512, 128, 64)), "rmem",
            ])
            return tf.schedule((acc, held), op=T.copy())


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 128)),
)
class WgmmaCall:
    drained = Drain

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
                acc = tf.schedule(
                    (acc, lhs, rhs),
                    op=T.tiled_mma(atom=T.cuda.sm90.Wgmma(
                        n=16, dtype="bf16", form=T.cuda.sm90.Form.SS,
                        a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K,
                    )),
                )
                return tf.insert_slice(output, drained(acc), (0, 0))  # noqa: F821


del Drain
