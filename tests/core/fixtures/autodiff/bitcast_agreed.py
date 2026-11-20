"""A Bitcast that keeps the layout states a value's copies are equal before a Broadcast reader.

x is Varying on dp; the Bitcast reads it as Broadcast (form ii). Its gradient is the
Broadcast cotangent read back as Varying on dp.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K = 4, 6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class BitcastAgreed:
    @func(mesh=Mesh(("gpu",), (2,), names=("dp",)))
    def forward(x: Tensor[(M, K), "f32", (M, K) @ {mesh.dp @ V()}, "gmem"], w: Tensor[(M, K), "f32", (M, K), "gmem"]):
        agreed = tf.bitcast(x, (M, K))
        return agreed * agreed * w
