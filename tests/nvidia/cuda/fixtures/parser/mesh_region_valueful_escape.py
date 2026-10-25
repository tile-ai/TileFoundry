"""A mesh region whose with result and escaping assignment are one value."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="valueful_escape",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2),),
)
class ValuefulEscape:
    @func
    def valueful_escape(x: Tensor[(2,), "f32"]):
        with Mesh(("cta",), layout=(2,), names=("tile",)) as _mesh:
            value = tf.relu(x)
            for _index in tf.tile(2, 1):  # noqa: F821
                value = tf.relu(value)
        return value
