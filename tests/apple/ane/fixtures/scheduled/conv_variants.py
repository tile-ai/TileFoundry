"""Five convolutions of one feature map, stored into channel windows of one output.

A 1x1 convolution with a bias and ReLU, a 1x1 one with the SiLU table, a 1x1 one with
the sigmoid table, a 3x3 one that zero-pads by one, whose result a pass-through
writes out from L2, and a 3x3 depthwise one; each has its own constant weights.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="conv_variants", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def conv_variants(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"],
        rectified: ConstTensor[(32, 32, 1, 1), "f16", ((32, 32, 1, 1), (32, 1, 1, 1)), "gmem"],
        b: ConstTensor[(32,), "f16", ((32,), (1,)), "gmem"],
        silu: ConstTensor[(32, 32, 1, 1), "f16", ((32, 32, 1, 1), (32, 1, 1, 1)), "gmem"],
        sigmoid: ConstTensor[(32, 32, 1, 1), "f16", ((32, 32, 1, 1), (32, 1, 1, 1)), "gmem"],
        padded: ConstTensor[(32, 32, 3, 3), "f16", ((32, 32, 3, 3), (288, 9, 3, 1)), "gmem"],
        depthwise: ConstTensor[(32, 1, 3, 3), "f16", ((32, 1, 3, 3), (9, 9, 3, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            out = tf.empty(Tensor[(1, 160, 16, 16), "f16", ((1, 160, 16, 16), (81920, 512, 32, 1)), "gmem"])
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            y = tf.schedule((xs, rectified, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"]), b), op=T.ane.conv(nonlinear=T.ane.Nonlinear.RELU))
            tile = tf.schedule((y, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (81920, 512, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 0, 0))
            y = tf.schedule((xs, silu, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.conv(nonlinear=T.ane.Nonlinear.SILU))
            tile = tf.schedule((y, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (81920, 512, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 32, 0, 0))
            y = tf.schedule((xs, sigmoid, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.conv(nonlinear=T.ane.Nonlinear.SIGMOID))
            tile = tf.schedule((y, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (81920, 512, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 64, 0, 0))
            h = tf.schedule((xs, padded, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.conv(padding=(1, 1)))
            y = tf.schedule((h, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.bypass())
            tile = tf.schedule((y, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (81920, 512, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 96, 0, 0))
            y = tf.schedule((xs, depthwise, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.conv(padding=(1, 1), groups=32))
            tile = tf.schedule((y, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (81920, 512, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 128, 0, 0))
            return out
