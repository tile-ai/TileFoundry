"""Two constant-weight convolutions sharing a declared output buffer."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="multi_weight", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def multi_weight(
        x: Tensor[(1, 32, 2, 32), "f16", ((1, 32, 2, 32), (2048, 64, 32, 1)), "gmem"],
        second: ConstTensor[(32, 32, 1, 1), "f16", ((32, 32, 1, 1), (32, 1, 1, 1)), "gmem"],
        first: ConstTensor[(32, 32, 1, 1), "f16", ((32, 32, 1, 1), (32, 1, 1, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            intermediate = tf.schedule(
                (tf.empty(Tensor[(1, 32, 2, 32), "f16", ((1, 32, 2, 32), (2048, 64, 32, 1)), "gmem"]), x, first),
                op=T.ane.conv(),
            )
            return tf.schedule(
                (intermediate, x, second),
                op=T.ane.conv(),
            )
