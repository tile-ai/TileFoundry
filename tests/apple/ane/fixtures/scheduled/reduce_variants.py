"""A sum, a mean and a sum of squares over both spatial axes, stored into three channel windows.

The mean, a sum scaled by 1/256, stays in L2 for a planar pass.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="reduce_variants", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def reduce_variants(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            out = tf.empty(Tensor[(1, 96, 1, 1), "f16", ((1, 96, 1, 1), (3072, 32, 32, 1)), "gmem"])
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            r = tf.schedule((xs, tf.empty(Tensor[(1, 32, 1, 1), "f16", ((1, 32, 1, 1), (256, 8, 8, 1)), "smem"])), op=T.ane.reduce(axes=(2, 3), keepdim=True, kind=ReduceKind.SUM))
            tile = tf.schedule((r, tf.empty(Tensor[(1, 32, 1, 1), "f16", ((1, 32, 1, 1), (3072, 32, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 0, 0))
            r = tf.schedule((xs, tf.empty(Tensor[(1, 32, 1, 1), "f16", ((1, 32, 1, 1), (256, 8, 8, 1)), "smem"])), op=T.ane.reduce(axes=(2, 3), keepdim=True, kind=ReduceKind.SUM, scale=1 / 256))
            y = tf.schedule((r, tf.empty(Tensor[(1, 32, 1, 1), "f16", ((1, 32, 1, 1), (256, 8, 8, 1)), "smem"])), op=T.ane.unary(scale=2.0, bias=1.0))
            tile = tf.schedule((y, tf.empty(Tensor[(1, 32, 1, 1), "f16", ((1, 32, 1, 1), (3072, 32, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 32, 0, 0))
            r = tf.schedule((xs, tf.empty(Tensor[(1, 32, 1, 1), "f16", ((1, 32, 1, 1), (256, 8, 8, 1)), "smem"])), op=T.ane.reduce(axes=(2, 3), keepdim=True, kind=ReduceKind.SUM, square=True))
            tile = tf.schedule((r, tf.empty(Tensor[(1, 32, 1, 1), "f16", ((1, 32, 1, 1), (3072, 32, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 64, 0, 0))
            return out
