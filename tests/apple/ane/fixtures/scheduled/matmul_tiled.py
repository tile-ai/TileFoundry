"""A [512, 16] @ [16, 16] matrix product scheduled in four 128-row tiles that share the weights.

Each tile is a window of 16 channels of the canonical maps of `matmul.py`: a load of
its rows, a swap, a convolution, a swap and a store into its rows of the output.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="matmul_tiled", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def matmul_tiled(
        a: Tensor[(1, 64, 8, 16), "f16", ((1, 64, 8, 16), (16384, 256, 32, 1)), "gmem"],
        w: ConstTensor[(16, 16, 1, 1), "f16", ((16, 16, 1, 1), (16, 1, 1, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            out = tf.empty(Tensor[(1, 64, 8, 16), "f16", ((1, 64, 8, 16), (16384, 256, 32, 1)), "gmem"])
            al = tf.schedule((a[:, 0:16, :, :], tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.load())
            outer = tf.schedule((al, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            y = tf.schedule((outer, w, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.conv())
            rows = tf.schedule((y, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            tile = tf.schedule((rows, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (16384, 256, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 0, 0))
            al = tf.schedule((a[:, 16:32, :, :], tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.load())
            outer = tf.schedule((al, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            y = tf.schedule((outer, w, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.conv())
            rows = tf.schedule((y, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            tile = tf.schedule((rows, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (16384, 256, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 16, 0, 0))
            al = tf.schedule((a[:, 32:48, :, :], tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.load())
            outer = tf.schedule((al, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            y = tf.schedule((outer, w, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.conv())
            rows = tf.schedule((y, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            tile = tf.schedule((rows, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (16384, 256, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 32, 0, 0))
            al = tf.schedule((a[:, 48:64, :, :], tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.load())
            outer = tf.schedule((al, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            y = tf.schedule((outer, w, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.conv())
            rows = tf.schedule((y, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (2048, 128, 16, 1)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            tile = tf.schedule((rows, tf.empty(Tensor[(1, 16, 8, 16), "f16", ((1, 16, 8, 16), (16384, 256, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 48, 0, 0))
            return out
