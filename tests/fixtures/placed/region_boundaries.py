"""Every legal region-boundary shape in one program, priced and round-tripped."""

from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CudaTarget


@module(
    entry="run",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class RegionBoundaries:
    @func
    def helper(x: Tensor[(8,), "f32"]):
        """A callee with its own thread region, called inside a CTA region."""
        with Mesh(("thread",), (4,), names=("t",)) as m:
            r = tf.reshard(x, (8 @ m.t,), "rmem")
            p = tf.reshard(r * 2.0, (8,), "gmem")
        return p

    @func
    def run(x: Tensor[(8,), "f32"]):
        with Mesh(("thread",), (4,), names=("t",)) as m:
            r = tf.reshard(x, (8 @ m.t,), "rmem")
            v2 = tf.reshard(r * 2.0, (8,), "gmem")
        with Mesh(("cta",), (2,), names=("c",)) as _c:
            a = helper(x)
            t = v2
            b = tf.reshard(t, (8,), "smem")
        with Mesh(("cta",), (2,), names=("c2",)) as _c2:
            with Mesh(("thread",), (4,), names=("t2",)) as m2:
                held = tf.reshard(v2, (8 @ m2.t2,), "rmem")
                y = tf.reshard(held + 1.0, (8,), "gmem")
                return a, b, y
