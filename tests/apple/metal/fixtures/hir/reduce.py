"""Two SIMD groups combine a per-lane Partial(sum); Metal has no bf16 SIMD reduction."""
from tilefoundry.dsl import *
from tilefoundry_apple import metal

SIMDGROUPS = Mesh((Topology("thread", 64),), layout=Layout(((2, 32),), ((32, 1),)))
PARTIAL_SUM = ShardLayout(Layout((2, 1), (0, 0)), (S(0), P("sum")), SIMDGROUPS)


@module(entry="hir_reduce", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def hir_reduce(
        single: Tensor[(2, 1), "f32", PARTIAL_SUM, "rmem"],
        brain: Tensor[(2, 1), "bf16", PARTIAL_SUM, "rmem"],
    ):
        with Mesh(("thread",), layout=Layout(((2, 32),), ((32, 1),))) as simd:
            total = tf.reshard(single, ShardLayout(Layout((2, 1), (0, 0)), (S(0), B()), simd), "rmem")
            brain_total = tf.reshard(brain, ShardLayout(Layout((2, 1), (0, 0)), (S(0), B()), simd), "rmem")
            return total, brain_total
