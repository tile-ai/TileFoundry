"""A retained callee returns a WGMMA accumulator before its group completes."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="issue")
class Issue:
    @func
    def issue(a: Tensor[(64, 16), "bf16"], b: Tensor[(16, 16), "bf16"]) -> Tensor[
        (64, 16), "f32", "rmem"
    ]:
        with Mesh(
            (Topology("thread", 128),), layout=(4, 8, 4), names=("warp", "lane8", "lane4")
        ) as threads:
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
            return tf.schedule(
                (acc, lhs, rhs),
                op=T.tiled_mma(atom=T.cuda.sm90.Wgmma(
                    n=16, dtype="bf16", form=T.cuda.sm90.Form.SS,
                    a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K,
                )),
            )


@module(
    entry="gemm",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 128)),
)
class WgmmaReturn:
    issued = Issue

    @func
    def gemm(
        a: Tensor[(64, 16), "bf16"], b: Tensor[(16, 16), "bf16"]
    ) -> Tensor[(64, 16), "f32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as _t:
                output = tf.zeros(Tensor[(64, 16), "f32"])
                return tf.insert_slice(output, issued(a, b), (0, 0))


del Issue
