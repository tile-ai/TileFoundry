"""One thread lands a row through cp.async and reads it back.

The thread's own program order needs no barrier, but its read still waits for
the group: a cp.async write is visible to the issuing thread only once the wait
returns.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="gemm",
    target=cuda.Target("nvidia.rtx_6000_ada"),
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class OneThreadCpAsync:
    @func
    def gemm(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as threads:
                with threads[0:1] as _writer:
                    stage = tf.empty(Tensor[(8,), "f32", (8,), "smem"])
                    landed = tf.schedule((x, stage), op=T.copy_async())
                    return tf.schedule((landed, tf.empty(Tensor[(8,), "f32"])), op=T.copy())
