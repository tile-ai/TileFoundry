"""A 3x3 convolution with bias on one NE task, from unified memory to unified memory."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="conv", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def conv(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"],
        w: Tensor[(32, 32, 3, 3), "f16", ((32, 32, 3, 3), (288, 9, 3, 1)), "gmem"],
        b: Tensor[(32,), "f16", ((32,), (1,)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"]), x, w, b),
                op=T.ane.conv(padding=(1, 1)),
            )
