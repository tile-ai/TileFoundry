"""A padded 3x3 average pool that divides by the cells each window holds inside the map."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="pool_avg", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def pool_avg(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]), x),
                op=T.ane.pool(
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), count_include_pad=False, kind=PoolKind.AVG
                ),
            )
