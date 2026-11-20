"""Sixteen quad groups of four lanes each sum their lanes."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

QUADS = Mesh((Topology("thread", 64),), layout=Layout(((16, 4),), ((4, 1),)))
PER_LANE = ShardLayout(Layout((16, 4), (0, 0)), (S(0), S(1)), QUADS)


@module(entry="quad_reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def quad_reduce(x: Tensor[(16, 4), "i32", PER_LANE, "rmem"]):
        with Mesh(("thread",), layout=Layout(((16, 4),), ((4, 1),))) as quads:
            shared = ShardLayout(Layout((16, 1), (0, 0)), (S(0), B()), quads)
            return tf.schedule(
                (tf.empty(Tensor[(16, 1), "i32", shared, "rmem"]), x),
                op=T.metal.reduce(kind=T.metal.ReduceKind.SUM, dtype=DType.i32),
            )
