"""An affine task over a feature map larger than L2, scheduled in eight 16-row tiles: x + 1."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="unary_tiled", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def unary_tiled(x: Tensor[(1, 64, 128, 128), "f16", ((1, 64, 128, 128), (1048576, 16384, 128, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            out = tf.empty(Tensor[(1, 64, 128, 128), "f16", ((1, 64, 128, 128), (1048576, 16384, 128, 1)), "gmem"])
            xs = tf.schedule((x[:, :, 0:16, :], tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.unary(bias=1.0))
            tile = tf.schedule((ys, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (1048576, 16384, 128, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 0, 0))
            xs = tf.schedule((x[:, :, 16:32, :], tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.unary(bias=1.0))
            tile = tf.schedule((ys, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (1048576, 16384, 128, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 16, 0))
            xs = tf.schedule((x[:, :, 32:48, :], tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.unary(bias=1.0))
            tile = tf.schedule((ys, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (1048576, 16384, 128, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 32, 0))
            xs = tf.schedule((x[:, :, 48:64, :], tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.unary(bias=1.0))
            tile = tf.schedule((ys, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (1048576, 16384, 128, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 48, 0))
            xs = tf.schedule((x[:, :, 64:80, :], tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.unary(bias=1.0))
            tile = tf.schedule((ys, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (1048576, 16384, 128, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 64, 0))
            xs = tf.schedule((x[:, :, 80:96, :], tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.unary(bias=1.0))
            tile = tf.schedule((ys, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (1048576, 16384, 128, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 80, 0))
            xs = tf.schedule((x[:, :, 96:112, :], tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.unary(bias=1.0))
            tile = tf.schedule((ys, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (1048576, 16384, 128, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 96, 0))
            xs = tf.schedule((x[:, :, 112:128, :], tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.load())
            ys = tf.schedule((xs, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (131072, 2048, 128, 1)), "smem"])), op=T.ane.unary(bias=1.0))
            tile = tf.schedule((ys, tf.empty(Tensor[(1, 64, 16, 128), "f16", ((1, 64, 16, 128), (1048576, 16384, 128, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 112, 0))
            return out
