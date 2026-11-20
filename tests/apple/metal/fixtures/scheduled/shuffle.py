"""Two SIMD groups rotate their lanes down by one, broadcast lane 5, and rotate u32 lanes up."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 32),), ((32, 1),)))
PER_LANE = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), SIMDGROUPS)
UNIFORM = ShardLayout(Layout((), ()), (B(),), SIMDGROUPS)


@module(entry="shuffle", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def shuffle(
        x: Tensor[(2, 32), "i32", PER_LANE, "rmem"],
        wide: Tensor[(2, 32), "u32", PER_LANE, "rmem"],
        one: Tensor[(), "u16", UNIFORM, "rmem"],
        five: Tensor[(), "u16", UNIFORM, "rmem"],
    ):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            lanes = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), simd)
            down = tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", lanes, "rmem"]), x, one),
                op=T.metal.shuffle(kind=T.metal.ShuffleKind.ROTATE_DOWN, dtype=DType.i32),
            )
            broadcast = tf.schedule(
                (tf.empty(Tensor[(2, 32), "i32", lanes, "rmem"]), x, five),
                op=T.metal.shuffle(kind=T.metal.ShuffleKind.BROADCAST, dtype=DType.i32),
            )
            up = tf.schedule(
                (tf.empty(Tensor[(2, 32), "u32", lanes, "rmem"]), wide, one),
                op=T.metal.shuffle(kind=T.metal.ShuffleKind.ROTATE_UP, dtype=DType.u32),
            )
            return down, broadcast, up
