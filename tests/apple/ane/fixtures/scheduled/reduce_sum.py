"""A sum over both spatial axes with a singleton output row and column."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="reduce_sum", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def reduce_sum(x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule(
                (tf.empty(Tensor[(1, 32, 1, 1), "f16", ((1, 32, 1, 1), (1024, 32, 32, 1)), "gmem"]), x),
                op=T.ane.reduce(axes=(2, 3), keepdim=True, kind=ReduceKind.SUM),
            )
