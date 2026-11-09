"""A root that calls the same child Expert twice and adds the results."""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.logical.hir_composition import Expert
from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

_H200 = "nvidia.h200_sxm"
_CTA = (Topology("cta", 132),)


@module(entry="twice", target=cuda.Target(_H200), topologies=_CTA)
class _Twice:
    mm = Expert

    @func
    def twice(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return tf.add(mm(x), mm(x))
