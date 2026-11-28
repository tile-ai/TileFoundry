"""Two matrix products ([128, 16] @ [16, 16]) @ [16, 16]; the first stays in the on-chip buffer.

The matrices are the canonical feature maps of `matmul.py`. Both convolutions read
their source in L2, so the first product never returns to unified memory.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="chain", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def chain(
        a: Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (4096, 256, 32, 1)), "gmem"],
        b: ConstTensor[(16, 16, 1, 1), "f16", ((16, 16, 1, 1), (16, 1, 1, 1)), "gmem"],
        w: ConstTensor[(16, 16, 1, 1), "f16", ((16, 16, 1, 1), (16, 1, 1, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            al = tf.schedule((a, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.load())
            outer = tf.schedule((al, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            h = tf.schedule((outer, b, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.conv())
            y = tf.schedule((h, w, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.conv())
            rows = tf.schedule((y, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            return tf.schedule((rows, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (4096, 256, 32, 1)), "gmem"])), op=T.ane.store())
