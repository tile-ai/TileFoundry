"""A matmul whose roofline bound is larger than float precision holds exactly."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_ROUNDING_M = 14_593
_ROUNDING_N = 11_489
_ROUNDING_K = 298_224_413
_H200 = CudaTarget("nvidia.h200_sxm")
_LARGE_H200 = CudaTarget(
    replace(_H200.device, hbm_capacity_bytes=300_000_000_000_000_000),
    architecture=_H200.architecture,
)


@module(entry="main", target=_LARGE_H200, topologies=(Topology("cta", 1),))
class _LargeRooflineRounding:
    @func
    def main(
        lhs: Tensor[(_ROUNDING_M, _ROUNDING_K), "f32"],
        rhs: ConstTensor[(_ROUNDING_K, _ROUNDING_N), "f32"],
    ):
        return tf.matmul(lhs, rhs)
