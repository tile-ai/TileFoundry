"""A Bitcast that keeps the layout states a value's copies are equal before a Broadcast reader."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K = 4, 6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class BitcastAgreed:
    @func(mesh=Mesh(("gpu",), (2,), names=("dp",)))
    def forward(x: Tensor[(M, K), "f32", "gmem"], w: Tensor[(M, K), "f32", (M, K) @ {B()}, "gmem"]):
        h = x * x
        agreed = tf.bitcast(h, (M, K) @ {B()})
        return agreed * w
