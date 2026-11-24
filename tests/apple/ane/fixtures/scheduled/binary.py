"""An elementwise sum of two feature maps on the processing engine."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="binary", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def binary(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"],
        y: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"]), x, y),
                op=T.ane.binary(kind=BinaryKind.ADD),
            )
