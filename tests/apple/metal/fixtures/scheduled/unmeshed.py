"""A product of two fragments outside any mesh; lowering refuses it."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),)))
FRAGMENT = ShardLayout(Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (S(0), S(4), S(2), S(3), S(1)), SIMDGROUPS)


@module(entry="unmeshed", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def unmeshed(
        a: Tensor[(2, 8, 8), "f32", FRAGMENT, "rmem"],
        b: Tensor[(2, 8, 8), "f32", FRAGMENT, "rmem"],
    ):
        return tf.matmul(a, b)
