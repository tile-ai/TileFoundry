"""A host function that launches a device function; ``launch`` is not an op call."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="host", target=CpuTarget(), topologies=(Topology("cta", 2),))
class LaunchHost:
    @func
    def device(a: Tensor[(2, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.mul(a, a)

    @prim_func(target=CpuTarget())
    def host(a: Tensor[(2, 4), "f32"], out: Tensor[(2, 4), "f32"]):
        launch(device, a, out, grid=(2, 1, 1), block=(1, 1, 1))  # noqa: F821
