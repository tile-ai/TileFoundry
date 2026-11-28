"""A [256, 16] @ [16, 32] matrix product as a load, a swap, a 1x1 convolution and a swap.

A row-major matrix with 64-byte rows is the canonical [1, M / 8, 8, W] feature map whose
row m is channel m // 8, row m % 8: load and store alone change rank. The first swap
of channels and width puts the contracted axis K on the channels and M / 8 on the
width, the convolution with the right operand as [N, K, 1, 1] weights contracts K
into N channels, and the second swap writes the product back row-major.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="matmul", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def matmul(
        a: Tensor[(256, 16), "f16", ((256, 16), (32, 1)), "gmem"],
        w: ConstTensor[(32, 16, 1, 1), "f16", ((32, 16, 1, 1), (16, 1, 1, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            al = tf.schedule((a, tf.empty(Tensor[(1, 32, 8, 16), "f16", ((1, 32, 8, 16), (4096, 128, 16, 1)), "smem"])), op=T.ane.load())
            outer = tf.schedule((al, tf.empty(Tensor[(1, 16, 8, 32), "f16", ((1, 16, 8, 32), (4096, 256, 32, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            y = tf.schedule((outer, w, tf.empty(Tensor[(1, 32, 8, 32), "f16", ((1, 32, 8, 32), (8192, 256, 32, 1)), "smem"])), op=T.ane.conv())
            rows = tf.schedule((y, tf.empty(Tensor[(1, 32, 8, 32), "f16", ((1, 32, 8, 32), (8192, 256, 32, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            return tf.schedule((rows, tf.empty(Tensor[(256, 32), "f16", ((256, 32), (32, 1)), "gmem"])), op=T.ane.store())
