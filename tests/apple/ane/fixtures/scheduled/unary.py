"""Four planar passes chained in L2 and a pass-through that writes out the last one.

The passes are an affine pass with a clamp, then abs with a clamp, scale and square.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="unary", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def unary(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            a = tf.schedule((xs, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.unary(pre_scale=2.0, bias=3.0, clamp=(0.0, 6.0)))
            b = tf.schedule((a, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.unary(bias=-2.0, abs=True, clamp=(0.0, 3.0)))
            c = tf.schedule((b, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.unary(scale=2.0, bias=1.0))
            d = tf.schedule((c, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.unary(bias=-5.0, square=True))
            y = tf.schedule((d, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.bypass())
            return tf.schedule((y, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"])), op=T.ane.store())
