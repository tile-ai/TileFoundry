"""A shared-memory tile written '@ warp' inside cta x a thread slice: Broadcast on that scope.

In a body, '@ warp' keeps the cta topology of the scope and takes warp's thread units,
so the tile holds the same type as Broadcast on the current composed scope.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="run", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 128)))
class ScopeNameLayout:
    @func
    def run(x: Tensor[(16, 16), "f32"]) -> Tensor[(16, 16), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(128,), names=("t",)) as threads:
                with threads[32:64] as warp:
                    tile = tf.schedule((x, tf.empty(Tensor[(16, 16), "f32", (16, 16) @ warp, "smem"])), op=T.copy())
                    return tf.schedule((tile, tf.empty(Tensor[(16, 16), "f32"])), op=T.copy())
