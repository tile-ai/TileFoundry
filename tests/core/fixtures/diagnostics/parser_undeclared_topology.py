"""A function mesh over a topology its module does not declare."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="run", target=CpuTarget())
class MissingTopology:
    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def run(x: Tensor[(8 @ mesh.b, 8), "f32"]):  # noqa: F821
        return tf.add(x, x)
