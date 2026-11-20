"""Quad groups of four lanes cannot shift runs of 8 lanes; verification refuses modulo > width."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

QUADS = Mesh((Topology("thread", 64),), layout=Layout(((16, 4),), ((4, 1),)))
PER_LANE = ShardLayout(Layout((16, 4), (0, 0)), (S(0), S(1)), QUADS)
UNIFORM = ShardLayout(Layout((), ()), (B(),), QUADS)


@module(entry="shuffle_fill_modulo", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def shuffle_fill_modulo(
        x: Tensor[(16, 4), "i32", PER_LANE, "rmem"],
        delta: Tensor[(), "u16", UNIFORM, "rmem"],
    ):
        with Mesh(("thread",), layout=Layout(((16, 4),), ((4, 1),))) as quads:
            lanes = ShardLayout(Layout((16, 4), (0, 0)), (S(0), S(1)), quads)
            return tf.schedule(
                (tf.empty(Tensor[(16, 4), "i32", lanes, "rmem"]), x, x, delta),
                op=T.metal.shuffle_fill(kind=T.metal.ShuffleKind.UP, modulo=8, dtype=DType.i32),
            )
