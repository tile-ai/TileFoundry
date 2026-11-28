"""Two convolutions change the channel count: the second reads the L2 buffer and narrows it."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="conv_conv_rect", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def conv_conv_rect(
        x: Tensor[(1, 16, 16, 16), "f16", ((1, 16, 16, 16), (8192, 512, 32, 1)), "gmem"],
        v: ConstTensor[(32, 16, 1, 1), "f16", ((32, 16, 1, 1), (16, 1, 1, 1)), "gmem"],
        w: ConstTensor[(16, 32, 1, 1), "f16", ((16, 32, 1, 1), (32, 1, 1, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 16, 16, 16), "f16", ((1, 16, 16, 16), (4096, 256, 16, 1)), "smem"])), op=T.ane.load())
            h = tf.schedule((xs, v, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.conv())
            y = tf.schedule((h, w, tf.empty(Tensor[(1, 16, 16, 16), "f16", ((1, 16, 16, 16), (4096, 256, 16, 1)), "smem"])), op=T.ane.conv())
            return tf.schedule((y, tf.empty(Tensor[(1, 16, 16, 16), "f16", ((1, 16, 16, 16), (8192, 512, 32, 1)), "gmem"])), op=T.ane.store())
