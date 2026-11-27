"""The same convolution with a ReLU applied to its result in the one NE task."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="conv_relu", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def conv_relu(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"],
        w: Tensor[(32, 32, 3, 3), "f16", ((32, 32, 3, 3), (288, 9, 3, 1)), "gmem"],
        b: Tensor[(32,), "f16", ((32,), (1,)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]), x, w, b),
                op=T.ane.conv(padding=(1, 1), nonlinear=T.ane.Nonlinear.RELU),
            )
