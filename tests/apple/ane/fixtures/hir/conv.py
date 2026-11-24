"""Convolutions offered to ANE: one NE task takes an fp16 convolution of a 3x3 kernel.

ANE refuses fp32 operands, dilation, a stride beyond the engine's strides and padding
at or beyond the kernel.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="hir_conv", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def hir_conv(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"],
        w: Tensor[(32, 32, 3, 3), "f16", ((32, 32, 3, 3), (288, 9, 3, 1)), "gmem"],
        b: Tensor[(32,), "f16", ((32,), (1,)), "gmem"],
        xs: Tensor[(1, 32, 16, 16), "f32", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"],
        ws: Tensor[(32, 32, 3, 3), "f32", ((32, 32, 3, 3), (288, 9, 3, 1)), "gmem"],
        bs: Tensor[(32,), "f32", ((32,), (1,)), "gmem"],
        wg: Tensor[(32, 16, 3, 3), "f16", ((32, 16, 3, 3), (144, 9, 3, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            same = tf.conv2d(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
            strided = tf.conv2d(x, w, b, stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1)
            grouped = tf.conv2d(x, wg, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=2)
            single = tf.conv2d(xs, ws, bs, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
            dilated = tf.conv2d(x, w, b, stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=1)
            wide_step = tf.conv2d(x, w, b, stride=(3, 3), padding=(1, 1), dilation=(1, 1), groups=1)
            wide_pad = tf.conv2d(x, w, b, stride=(1, 1), padding=(3, 3), dilation=(1, 1), groups=1)
            return same, strided, grouped, single, dilated, wide_step, wide_pad
