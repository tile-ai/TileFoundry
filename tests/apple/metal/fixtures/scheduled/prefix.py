"""Two SIMD groups take the exclusive prefix sum of i32 lanes and the inclusive one of u32 lanes."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 32),), ((32, 1),)))
PER_LANE = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), SIMDGROUPS)


@module(entry="prefix", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def prefix(
        x: Tensor[(2, 32), "i32", PER_LANE, "rmem"],
        wide: Tensor[(2, 32), "u32", PER_LANE, "rmem"],
    ):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            lanes = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), simd)
            exclusive = tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", lanes, "rmem"]), x),
                op=T.metal.prefix(kind=T.metal.PrefixKind.SUM, inclusive=False, dtype=DType.i32),
            )
            inclusive = tf.schedule(
                (tf.empty(Tensor[(2, 32), "u32", lanes, "rmem"]), wide),
                op=T.metal.prefix(kind=T.metal.PrefixKind.SUM, inclusive=True, dtype=DType.u32),
            )
            return exclusive, inclusive
