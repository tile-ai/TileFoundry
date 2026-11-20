"""A split value resharded onto mesh slices that share a unit with its own."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=4)
gpu = Topology("gpu", 4)


@module(entry="run", target=target, topologies=(gpu,))
class SplitOverlap:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4 @ mesh[0:2].g, 8), "gmem"]):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as units:
            return tf.reshard(x, (4 @ units[1:3].g, 8), "gmem")
