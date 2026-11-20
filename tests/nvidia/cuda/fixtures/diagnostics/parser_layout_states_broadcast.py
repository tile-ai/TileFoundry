"""A layout that writes '@ {B()}': Broadcast is the default, so it is not written."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="run", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class BroadcastWritten:
    @func
    def run(x: Tensor[(4, 8), "f32"]):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _g:
            return tf.reshard(x, (4, 8) @ {B()}, "gmem")
