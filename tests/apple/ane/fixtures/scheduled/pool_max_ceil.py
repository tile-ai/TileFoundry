"""A 3x3 max pool with stride 2 and ceil mode whose last window row and column overhang the map."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="pool_max_ceil", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def pool_max_ceil(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            p = tf.schedule((xs, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (2048, 64, 8, 1)), "smem"])), op=T.ane.pool(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True, kind=PoolKind.MAX))
            return tf.schedule((p, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (8192, 256, 32, 1)), "gmem"])), op=T.ane.store())
