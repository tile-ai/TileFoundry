"""A thread-placed value consumed after escaping into a CTA-only scope."""

from tilefoundry import module
from tilefoundry.dsl import *


@module(
    entry="run",
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class ValueEscapes:
    @func
    def run(x: Tensor[(8,), "f32"]):
        with Mesh(("thread",), (4,), names=("lane",)) as lanes:
            value = tf.reshard(x, (8 @ lanes.lane,), "rmem")
        with Mesh(("cta",), (2,), names=("block",)) as _blocks:
            return tf.add(value, value)
