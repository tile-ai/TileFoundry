"""A convolution of an fp32 feature map; ANE refuses fp32 operands."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="wide", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def wide(
        x: Tensor[(1, 32, 16, 16), "f32", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"],
        w: Tensor[(32, 32, 3, 3), "f16", ((32, 32, 3, 3), (288, 9, 3, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"]), x, w),
                op=T.ane.conv(padding=(1, 1)),
            )
