from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def wrong_body(x: Tensor[(4,), "f32"]):
    return tf.zeros(Tensor[(4,), "f32", None, "tmem"])
