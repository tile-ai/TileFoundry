"""A Broadcast value of one mesh slice read in a scope of the other slice."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="run", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class BroadcastOtherSlice:
    @func
    def run(x: Tensor[(8,), "f32"]):
        with Mesh(("gpu",), (2,), names=("pp",)) as stages:
            with Mesh(stages[0], (1,), names=("s",)) as _first:
                held = tf.reshard(x, (8,) @ _first, "gmem")
            with Mesh(stages[1], (1,), names=("s",)) as _second:
                return tf.add(held, held)
