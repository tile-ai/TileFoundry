"""A 3x3 convolution that keeps only the windows lying wholly inside its source."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="conv_3x3_valid", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def conv_3x3_valid(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"],
        w: ConstTensor[(32, 32, 3, 3), "f16", ((32, 32, 3, 3), (288, 9, 3, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            y = tf.schedule((xs, w, tf.empty(Tensor[(1, 32, 14, 14), "f16", ((1, 32, 14, 14), (7168, 224, 16, 1)), "smem"])), op=T.ane.conv())
            return tf.schedule((y, tf.empty(Tensor[(1, 32, 14, 14), "f16", ((1, 32, 14, 14), (14336, 448, 32, 1)), "gmem"])), op=T.ane.store())
