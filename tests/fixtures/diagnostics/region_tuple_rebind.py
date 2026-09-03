"""A tuple cannot carry an outer region directly into another region body."""

from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf


@module(
    entry="run",
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class RegionTupleRebind:
    @func
    def run(x: Tensor[(8,), "f32"]):
        with Mesh(("thread",), (4,), names=("t",)) as m:
            r = tf.reshard(x, (8 @ m.t,), "rmem")
            v2 = tf.reshard(r * 2.0, (8,), "gmem")
        with Mesh(("cta",), (2,), names=("c",)) as _c:
            w = tf.reshard(x, (8,), "smem")
            return v2, w
