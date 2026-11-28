"""Three planar passes retain their intermediates in the engine's L2 buffer."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="unary_chain3", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def unary_chain3(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            h = tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"]), x),
                op=T.ane.unary(abs=True),
            )
            g = tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"]), h),
                op=T.ane.unary(bias=-2.0),
            )
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]), g),
                op=T.ane.unary(scale=2.0, bias=1.0),
            )
