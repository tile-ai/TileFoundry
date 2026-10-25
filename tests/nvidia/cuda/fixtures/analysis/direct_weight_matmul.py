"""The matmul of a constant weight, written in the root without a child call."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_H200 = "nvidia.h200_sxm"
_CTA = (Topology("cta", 132),)


@module(entry="direct", target=CudaTarget(_H200), topologies=_CTA)
class _Direct:
    @func
    def direct(
        x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 8), "f32"]
    ) -> Tensor[(4, 8), "f32"]:
        return tf.matmul(x, w)
