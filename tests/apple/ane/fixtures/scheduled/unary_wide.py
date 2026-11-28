"""An affine task over a feature map wider than the base patch: 2 * x + 1."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="unary_wide", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def unary_wide(x: Tensor[(1, 32, 16, 64), "f16", ((1, 32, 16, 64), (32768, 1024, 64, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 64), "f16", ((1, 32, 16, 64), (32768, 1024, 64, 1)), "smem"])), op=T.ane.load())
            y = tf.schedule((xs, tf.empty(Tensor[(1, 32, 16, 64), "f16", ((1, 32, 16, 64), (32768, 1024, 64, 1)), "smem"])), op=T.ane.unary(pre_scale=2.0, bias=1.0))
            return tf.schedule((y, tf.empty(Tensor[(1, 32, 16, 64), "f16", ((1, 32, 16, 64), (32768, 1024, 64, 1)), "gmem"])), op=T.ane.store())
