"""A convolution reads the channel and width swap a pass-through task left in the L2 buffer.

This is the two-task shape of a matrix product: the swap puts the contracted axis
on the channels, and a 1x1 convolution contracts it.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="transpose_conv_chain", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def transpose_conv_chain(
        x: Tensor[(1, 32, 8, 16), "f16", ((1, 32, 8, 16), (8192, 256, 32, 1)), "gmem"],
        w: ConstTensor[(16, 16, 1, 1), "f16", ((16, 16, 1, 1), (16, 1, 1, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            h = tf.schedule(
                (tf.empty(Tensor[(1, 16, 8, 32), "f16", ((1, 16, 8, 32), (4096, 256, 32, 1)), "smem"]), x),
                op=T.ane.bypass(perm=(2, 1, 0)),
            )
            return tf.schedule(
                (tf.empty(Tensor[(1, 16, 8, 32), "f16", ((1, 16, 8, 32), (4096, 256, 32, 1)), "gmem"]), h, w),
                op=T.ane.conv(),
            )
