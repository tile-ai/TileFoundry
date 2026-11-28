"""A single-task 1x1 convolution that adds a constant bias to each output channel."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="conv_1x1_bias", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def conv_1x1_bias(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"],
        w: ConstTensor[(32, 32, 1, 1), "f16", ((32, 32, 1, 1), (32, 1, 1, 1)), "gmem"],
        b: ConstTensor[(32,), "f16", ((32,), (1,)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]), x, w, b),
                op=T.ane.conv(),
            )
