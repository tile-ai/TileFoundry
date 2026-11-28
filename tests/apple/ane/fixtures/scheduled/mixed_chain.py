"""Neural and planar tasks that read each other's results in L2, stored into two 14x14 windows.

A pass-through and a planar pass lead to a 1x1 convolution from 16 to 32 channels;
a planar pass reads its result and a planar sum adds both. A 3x3 convolution
without padding narrows the sum and the 1x1 result back to 16 channels, each
written out into its window.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="mixed_chain", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def mixed_chain(
        x: Tensor[(1, 16, 16, 16), "f16", ((1, 16, 16, 16), (8192, 512, 32, 1)), "gmem"],
        v: ConstTensor[(32, 16, 1, 1), "f16", ((32, 16, 1, 1), (16, 1, 1, 1)), "gmem"],
        w: ConstTensor[(16, 32, 3, 3), "f16", ((16, 32, 3, 3), (288, 9, 3, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            out = tf.empty(Tensor[(1, 32, 14, 14), "f16", ((1, 32, 14, 14), (14336, 448, 32, 1)), "gmem"])
            xs = tf.schedule((x, tf.empty(Tensor[(1, 16, 16, 16), "f16", ((1, 16, 16, 16), (4096, 256, 16, 1)), "smem"])), op=T.ane.load())
            b = tf.schedule((xs, tf.empty(Tensor[(1, 16, 16, 16), "f16", ((1, 16, 16, 16), (4096, 256, 16, 1)), "smem"])), op=T.ane.bypass())
            u = tf.schedule((b, tf.empty(Tensor[(1, 16, 16, 16), "f16", ((1, 16, 16, 16), (4096, 256, 16, 1)), "smem"])), op=T.ane.unary(scale=2.0, bias=1.0))
            h = tf.schedule((u, v, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.conv())
            p = tf.schedule((h, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.unary(abs=True))
            s = tf.schedule((h, p, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.binary(kind=BinaryKind.ADD))
            y = tf.schedule((s, w, tf.empty(Tensor[(1, 16, 14, 14), "f16", ((1, 16, 14, 14), (3584, 224, 16, 1)), "smem"])), op=T.ane.conv())
            tile = tf.schedule((y, tf.empty(Tensor[(1, 16, 14, 14), "f16", ((1, 16, 14, 14), (14336, 448, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 0, 0))
            y = tf.schedule((h, w, tf.empty(Tensor[(1, 16, 14, 14), "f16", ((1, 16, 14, 14), (3584, 224, 16, 1)), "smem"])), op=T.ane.conv())
            tile = tf.schedule((y, tf.empty(Tensor[(1, 16, 14, 14), "f16", ((1, 16, 14, 14), (14336, 448, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 16, 0, 0))
            return out
