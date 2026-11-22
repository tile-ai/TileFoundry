"""A pass-through task swaps the channel and width axes of a feature map."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="transpose", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def transpose(x: Tensor[(1, 32, 8, 16), "f16", ((1, 32, 8, 16), (4096, 128, 16, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 16, 8, 32), "f16", ((1, 16, 8, 32), (4096, 256, 32, 1)), "gmem"]), x),
                op=T.ane.bypass(perm=(2, 1, 0)),
            )
