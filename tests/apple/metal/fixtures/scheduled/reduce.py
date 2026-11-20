"""Two SIMD groups reduce i32 lanes (sum, max) and u32 lanes (max), and resolve a Partial(sum)."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 32),), ((32, 1),)))
PER_LANE = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), SIMDGROUPS)
PARTIAL_SUM = ShardLayout(Layout((2, 1), (0, 0)), (S(0), P("sum")), SIMDGROUPS)


@module(entry="reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def reduce(
        x: Tensor[(2, 32), "i32", PER_LANE, "rmem"],
        wide: Tensor[(2, 32), "u32", PER_LANE, "rmem"],
        partial: Tensor[(2, 1), "f32", PARTIAL_SUM, "rmem"],
    ):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            shared = ShardLayout(Layout((2, 1), (0, 0)), (S(0), B()), simd)
            total = tf.schedule(
                (tf.empty(Tensor[(2, 1), "i32", shared, "rmem"]), x),
                op=T.metal.reduce(kind=T.metal.ReduceKind.SUM, dtype=DType.i32),
            )
            largest = tf.schedule(
                (tf.empty(Tensor[(2, 1), "i32", shared, "rmem"]), x),
                op=T.metal.reduce(kind=T.metal.ReduceKind.MAX, dtype=DType.i32),
            )
            wide_largest = tf.schedule(
                (tf.empty(Tensor[(2, 1), "u32", shared, "rmem"]), wide),
                op=T.metal.reduce(kind=T.metal.ReduceKind.MAX, dtype=DType.u32),
            )
            resolved = tf.schedule((tf.empty(Tensor[(2, 1), "f32", shared, "rmem"]), partial), op=T.metal.reduce())
            return total, largest, wide_largest, resolved
