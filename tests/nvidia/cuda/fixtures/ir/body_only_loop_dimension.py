"""A loop that scans a context length the signature never mentions."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_LOOP_CTX = DimVar("loop_ctx", 1, 4096)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _LoopOnly:
    """A context length that appears nowhere in the signature.

    The loop scans it. Nothing about the parameters or the return says how long
    it is, which is what makes this the case a signature-driven rebuild misses.
    """

    @func
    def main(x: Tensor[(8,), "f32"]):
        total = tf.zeros(Tensor[(8,), "f32"])
        for _ in range(_LOOP_CTX):
            total = tf.add(total, x)
        return total
