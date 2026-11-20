"""A function mesh over a topology its module does not declare."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="run", target=cpu.Target())
class MissingTopology:
    mesh = Mesh(("cta",), (8,), names=("b",))

    @func
    def run(x: Tensor[(8 @ mesh.b, 8), "f32"]):
        with mesh:
            return tf.add(x, x)
