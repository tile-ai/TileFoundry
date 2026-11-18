"""Two SIMD groups move one 8x8 matrix each between device memory and registers."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),)))
FRAGMENT = ShardLayout(Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (S(0), S(4), S(2), S(3), S(1)), SIMDGROUPS)


@module(entry="hir_memory", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def hir_memory(
        memory: Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"],
        registers: Tensor[(2, 8, 8), "f32", FRAGMENT, "rmem"],
    ):
        with Mesh(("thread",), layout=Layout(((2, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),))) as simd:
            load = tf.reshard(
                memory,
                ShardLayout(Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (S(0), S(4), S(2), S(3), S(1)), simd),
                "rmem",
            )
            store = tf.reshard(registers, ((2, 8, 8), (64, 8, 1)), "gmem")
            return load, store
