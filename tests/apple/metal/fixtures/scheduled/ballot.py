"""Two SIMD groups vote on a predicate; bit t of the result is lane t's vote."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 32),), ((32, 1),)))
PER_LANE = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), SIMDGROUPS)


@module(entry="ballot", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def ballot(even: Tensor[(2, 32), "bool", PER_LANE, "rmem"]):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            shared = ShardLayout(Layout((2, 1), (0, 0)), (S(0), B()), simd)
            return tf.schedule((tf.empty(Tensor[(2, 1), "u32", shared, "rmem"]), even), op=T.metal.ballot())
