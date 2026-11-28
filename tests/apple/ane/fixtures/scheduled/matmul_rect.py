"""A [128, 16] @ [16, 32] matrix product: the convolution widens the channels from K to N."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="matmul_rect", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def matmul_rect(
        a: Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (4096, 256, 32, 1)), "gmem"],
        w: ConstTensor[(32, 16, 1, 1), "f16", ((32, 16, 1, 1), (16, 1, 1, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            al = tf.schedule((a, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.load())
            outer = tf.schedule((al, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            y = tf.schedule((outer, w, tf.empty(Tensor[(1, 32, 8, 16), "f16", ((1, 32, 8, 16), (4096, 128, 16, 1)), "smem"])), op=T.ane.conv())
            rows = tf.schedule((y, tf.empty(Tensor[(1, 16, 8, 32), "f16", ((1, 16, 8, 32), (4096, 256, 32, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            return tf.schedule((rows, tf.empty(Tensor[(1, 16, 8, 32), "f16", ((1, 16, 8, 32), (4096, 256, 32, 1)), "gmem"])), op=T.ane.store())
