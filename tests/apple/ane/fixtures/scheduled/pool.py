"""A padded, strided average pool on the processing engine."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="pool", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def pool(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 8, 8), "f16", ((1, 32, 8, 8), (8192, 256, 32, 1)), "gmem"]), x),
                op=T.ane.pool(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), kind=PoolKind.AVG),
            )
