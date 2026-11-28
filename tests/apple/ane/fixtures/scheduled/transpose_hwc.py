"""A pass-through task moves the channels of a feature map to the innermost axis."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="transpose_hwc", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def transpose_hwc(x: Tensor[(1, 32, 8, 16), "f16", ((1, 32, 8, 16), (8192, 256, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 8, 16), "f16", ((1, 32, 8, 16), (4096, 128, 16, 1)), "smem"])), op=T.ane.load())
            t = tf.schedule((xs, tf.empty(Tensor[(1, 8, 16, 32), "f16", ((1, 8, 16, 32), (4096, 512, 32, 1)), "smem"])), op=T.ane.bypass(perm=(1, 2, 0)))
            return tf.schedule((t, tf.empty(Tensor[(1, 8, 16, 32), "f16", ((1, 8, 16, 32), (4096, 512, 32, 1)), "gmem"])), op=T.ane.store())
