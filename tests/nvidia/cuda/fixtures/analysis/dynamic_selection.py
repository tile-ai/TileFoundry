"""A captured dynamic view that selects one of two shared-memory leaves."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="choose",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class DynamicSelection:
    @func
    def choose(k: Tensor[(), "i32"]):
        with Mesh(("cta",), (1,), names=("block",)) as cta:
            first = tf.zeros(Tensor[(4,), "bf16", (4,), "smem"])
            second = tf.zeros(Tensor[(4,), "bf16", (4,), "smem"])
            choices = (first, second)
            selected = choices[k]
            with cta[:] as _inner:
                picked = selected
            result = tf.add(picked, picked)
            return result
