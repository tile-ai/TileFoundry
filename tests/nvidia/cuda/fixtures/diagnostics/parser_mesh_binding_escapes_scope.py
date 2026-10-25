from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="escaped_mesh",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2),),
)
class EscapedMesh:
    @func
    def escaped_mesh(x: Tensor[(2,), "f32"]):
        with Mesh(("cta",), layout=(2,), names=("tile",)) as mesh:
            value = tf.relu(x)
        return tf.reshard(value, (2 @ mesh.tile,), "gmem")
