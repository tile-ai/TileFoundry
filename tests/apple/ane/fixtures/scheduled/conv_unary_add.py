"""A convolution result and a planar result of the same source added from L2."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="conv_unary_add", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def conv_unary_add(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"],
        w: ConstTensor[(32, 32, 1, 1), "f16", ((32, 32, 1, 1), (32, 1, 1, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            c = tf.schedule((xs, w, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.conv())
            u = tf.schedule((xs, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.unary(bias=-2.0))
            s = tf.schedule((c, u, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.binary(kind=BinaryKind.ADD))
            return tf.schedule((s, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"])), op=T.ane.store())
