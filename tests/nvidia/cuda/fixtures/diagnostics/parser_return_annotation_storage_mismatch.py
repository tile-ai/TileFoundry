from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def annotated(
    x: Tensor[(8, 16), "f32"],
) -> Tensor[(8, 16), "f32", None, "smem"]:
    return tf.mul(x, x)
