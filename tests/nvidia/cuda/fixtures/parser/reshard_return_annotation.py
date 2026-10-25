"""A return annotation that checks the layout a reshard in the body infers."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 2),))
def annotated(
    x: Tensor[(8, 16), "f32"],
) -> Tensor[(8, 16), "f32"]:
    with Mesh(("cta",), layout=(2,), names=("lane",)) as cta:
        return tf.reshard(x, (8 @ cta.lane, 16), "gmem")
