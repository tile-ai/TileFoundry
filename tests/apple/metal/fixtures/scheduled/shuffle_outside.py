"""Every lane of two SIMD groups names lane 40 as its source; evaluation refuses it."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 32),), ((32, 1),)))
PER_LANE = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), SIMDGROUPS)


@module(entry="shuffle_outside", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def shuffle_outside(
        x: Tensor[(2, 32), "i32", PER_LANE, "rmem"],
        lane: Tensor[(2, 32), "u16", PER_LANE, "rmem"],
    ):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            lanes = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), simd)
            return tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", lanes, "rmem"]), x, lane),
                op=T.metal.shuffle(kind=T.metal.ShuffleKind.INDEX, dtype=DType.i32),
            )
