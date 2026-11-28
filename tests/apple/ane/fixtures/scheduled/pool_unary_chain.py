"""A pool result read by a planar pass from L2: no Apple stream shows it, so it is refused."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="pool_unary_chain", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def pool_unary_chain(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            xs = tf.schedule((x, tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"])), op=T.ane.load())
            p = tf.schedule((xs, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (2048, 64, 8, 1)), "smem"])), op=T.ane.pool(kernel_size=(2, 2), stride=(2, 2), kind=PoolKind.MAX))
            y = tf.schedule((p, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (2048, 64, 8, 1)), "smem"])), op=T.ane.unary(scale=2.0, bias=1.0))
            return tf.schedule((y, tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (8192, 256, 32, 1)), "gmem"])), op=T.ane.store())
