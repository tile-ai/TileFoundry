"""Each lane stages its own row through shared memory, in a loop whose trip count is its lane.

A lane reads back only the bytes it wrote, so it needs no barrier; none could
sit in that loop. ``staged_dynamic_inner`` is the same loop with every lane on
one row, which is refused.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="gemm",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class LaneRows:
    @func
    def gemm(x: Tensor[(32, 8), "bf16"]) -> Tensor[(32, 8), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as threads:
                held = tf.schedule(
                    (x, tf.empty(Tensor[(32, 8), "bf16", ((32 @ threads.lane, 8), (0, 1)), "rmem"])),
                    op=T.copy(),
                )
                for _i in range(2):
                    for _j in range(threads.lane):
                        row = tf.schedule(
                            (
                                held,
                                tf.empty(
                                    Tensor[(32, 8), "bf16", ((32 @ threads.lane, 8), (8, 1)), "smem"]
                                ),
                            ),
                            op=T.copy(),
                        )
                        held = tf.schedule(
                            (
                                row,
                                tf.empty(
                                    Tensor[(32, 8), "bf16", ((32 @ threads.lane, 8), (0, 1)), "rmem"]
                                ),
                            ),
                            op=T.copy(),
                        )
                output = tf.zeros(Tensor[(32, 8), "bf16"])
                return tf.insert_slice(output, held, (0, 0))
