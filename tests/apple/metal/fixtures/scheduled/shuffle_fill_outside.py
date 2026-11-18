"""Two SIMD groups shift each run of 8 lanes up by 9 lanes; evaluation refuses it."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 32),), ((32, 1),)))
PER_LANE = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), SIMDGROUPS)
UNIFORM = ShardLayout(Layout((), ()), (B(),), SIMDGROUPS)


@module(entry="shuffle_fill_outside", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def shuffle_fill_outside(
        x: Tensor[(2, 32), "i32", PER_LANE, "rmem"],
        fill: Tensor[(2, 32), "i32", PER_LANE, "rmem"],
        delta: Tensor[(), "u16", UNIFORM, "rmem"],
    ):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            lanes = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), simd)
            return tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", lanes, "rmem"]), x, fill, delta),
                op=T.metal.shuffle_fill(kind=T.metal.ShuffleKind.UP, modulo=8, dtype=DType.i32),
            )
