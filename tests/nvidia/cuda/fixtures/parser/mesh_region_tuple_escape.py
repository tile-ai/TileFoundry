"""A value-less mesh region that two live values escape."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="multi_escape",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2),),
)
class MultiEscape:
    @func
    def multi_escape(x: Tensor[(2,), "f32"]):
        with Mesh(("cta",), layout=(2,), names=("tile",)) as _mesh:
            first = tf.relu(x)
            second = tf.relu(first)
        return first, second
