"""A CTA-placed register value consumed by a finer thread scope."""

from __future__ import annotations

from tilefoundry import module
from tilefoundry.dsl import *


@module(
    entry="run",
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class CoarseRmem:
    @func(mesh=Mesh(("cta",), (2,), names=("block",)))
    def run(x: Tensor[(8 @ mesh.block,), "f32", "rmem"]):  # noqa: F821
        with Mesh(("thread",), (4,), names=("lane",)) as _lanes:
            return tf.add(x, x)
