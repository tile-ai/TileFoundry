"""Five pools of one feature map, stored into channel windows of one 8x8 output.

A 2x2 max pool stays in L2, where a planar pass and a padded 3x3 average pool with
stride 1 read it. A padded max pool, a ceil-mode max pool and a ceil-mode average
pool read the loaded map. Each average divides by the cells its window holds
inside the map.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="pool_variants", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def pool_variants(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            out = tf.empty(Tensor[(1, 160, 8, 8), "f16", ((1, 160, 8, 8), (40960, 256, 32, 1)), "gmem"])
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            m = tf.schedule((xs, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (2048, 64, 8, 1)), "smem"])), op=T.ane.pool(kernel_size=(2, 2), stride=(2, 2), kind=PoolKind.MAX))
            y = tf.schedule((m, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (2048, 64, 8, 1)), "smem"])), op=T.ane.unary(scale=2.0, bias=1.0))
            tile = tf.schedule((y, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (40960, 256, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 0, 0, 0))
            p = tf.schedule((m, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (2048, 64, 8, 1)), "smem"])), op=T.ane.pool(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), count_include_pad=False, kind=PoolKind.AVG))
            tile = tf.schedule((p, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (40960, 256, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 32, 0, 0))
            p = tf.schedule((xs, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (2048, 64, 8, 1)), "smem"])), op=T.ane.pool(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), kind=PoolKind.MAX))
            tile = tf.schedule((p, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (40960, 256, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 64, 0, 0))
            p = tf.schedule((xs, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (2048, 64, 8, 1)), "smem"])), op=T.ane.pool(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True, kind=PoolKind.MAX))
            tile = tf.schedule((p, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (40960, 256, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 96, 0, 0))
            p = tf.schedule((xs, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (2048, 64, 8, 1)), "smem"])), op=T.ane.pool(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True, count_include_pad=False, kind=PoolKind.AVG))
            tile = tf.schedule((p, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (40960, 256, 32, 1)), "gmem"])), op=T.ane.store())
            out = tf.insert_slice(out, tile, (0, 128, 0, 0))
            return out
