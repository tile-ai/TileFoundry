"""A pass-through task loads a feature map into the engine's L2 buffer for a planar pass."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="bypass_unary_chain", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def bypass_unary_chain(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            h = tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"]), x),
                op=T.ane.bypass(),
            )
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]), h),
                op=T.ane.unary(scale=2.0, bias=1.0),
            )
