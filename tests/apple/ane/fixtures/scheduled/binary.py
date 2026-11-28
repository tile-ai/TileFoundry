"""Planar sums, a difference and a product of two feature maps, stored into three windows.

The sum and the difference read both loaded maps; the difference is written out. A
second sum and the product read two planar results from L2, one of which three
tasks read; the product stays in L2 for a last planar pass.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="binary", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def binary(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"],
        y: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            out = tf.empty(Tensor[(1, 96, 16, 16), "f16", ((1, 96, 16, 16), (49152, 512, 32, 1)), "gmem"])
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((y, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            s = tf.schedule((xs, ys, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.binary(kind=BinaryKind.ADD))
            d = tf.schedule((ys, xs, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.binary(kind=BinaryKind.SUB))
            tile = tf.schedule((d, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (49152, 512, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 0, 0))
            h = tf.schedule((s, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.unary(abs=True))
            g = tf.schedule((h, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.unary(bias=-2.0))
            a = tf.schedule((h, g, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.binary(kind=BinaryKind.ADD))
            tile = tf.schedule((a, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (49152, 512, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 32, 0, 0))
            p = tf.schedule((h, g, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.binary(kind=BinaryKind.MUL))
            q = tf.schedule((p, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.unary(scale=2.0, bias=1.0))
            tile = tf.schedule((q, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (49152, 512, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 64, 0, 0))
            return out
