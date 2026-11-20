"""Two SIMD groups select core's generic T.reduce, which Metal does not lower."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 32),), ((32, 1),)))
PER_LANE = ShardLayout(Layout((2, 32), (0, 0)), (S(0), S(1)), SIMDGROUPS)


@module(entry="core_reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def core_reduce(x: Tensor[(2, 32), "f32", PER_LANE, "rmem"]):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            total = tf.empty(Tensor[(2, 1), "f32", ShardLayout(Layout((2, 1), (0, 0)), (S(0), B()), simd), "rmem"])
            return tf.schedule((x, total), op=T.reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM))
