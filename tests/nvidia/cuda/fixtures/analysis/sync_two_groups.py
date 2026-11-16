"""One copy waits for a ``cp.async`` group and a WGMMA group at the same position.

``landed`` is still in flight from ``cp.async`` and ``acc`` from the WGMMA when
the InsertSlice copies the accumulator into the landed tile's bytes, so both
groups are committed and waited on before it, then a barrier runs. Lowering
emits an InsertSlice's base before its update, so the ``cp.async`` comes first.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="gemm",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 128)),
)
class TwoGroups:
    @func
    def gemm(
        a: Tensor[(64, 16), "bf16"], b: Tensor[(16, 16), "bf16"], x: Tensor[(64, 16), "f32"]
    ) -> Tensor[(64, 16), "f32", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(4, 8, 4), names=("warp", "lane8", "lane4")) as threads:
                output = tf.zeros(Tensor[(64, 16), "f32"])
                acc = tf.zeros(Tensor[
                    (64, 16), "f32",
                    ((4 @ threads.warp, 2, 8 @ threads.lane8, 2, 4 @ threads.lane4, 2,),
                     (16, 8, 1, 512, 128, 64)), "rmem",
                ])
                with threads[0, 0, 0] as _writer:
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
                    landed = tf.schedule(
                        (x, tf.empty(Tensor[(64, 16), "f32", Layout((64, 16), (16, 1)), "smem"])),
                        op=T.copy_async(),
                    )
                acc = tf.schedule(
                    (acc, lhs, rhs),
                    op=T.tiled_mma(atom=T.cuda.sm90.Wgmma(
                        n=16, dtype="bf16", form=T.cuda.sm90.Form.SS,
                        a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K,
                    )),
                )
                spilled = tf.insert_slice(landed, acc, (0, 0))
                return tf.insert_slice(output, spilled, (0, 0))
