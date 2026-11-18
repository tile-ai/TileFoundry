"""8x8 matrix products of two SIMD groups on fragments loaded from device memory.

Metal has no integer matrix product, and a product of device-memory operands
is no SIMD-group instruction.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import metal


@module(entry="hir_matmul", target=metal.Target("apple.m2_pro.gpu"),
        topologies=(Topology("threadgroup", 1), Topology("thread", 64)))
class Program:
    @func
    def hir_matmul(
        a: Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"],
        b: Tensor[(2, 8, 8), "f32", ((2, 8, 8), (64, 8, 1)), "gmem"],
        ai: Tensor[(2, 8, 8), "i32", ((2, 8, 8), (64, 8, 1)), "gmem"],
        bi: Tensor[(2, 8, 8), "i32", ((2, 8, 8), (64, 8, 1)), "gmem"],
    ):
        with Mesh(("thread",), layout=Layout(((2, 2, 4, 2, 2),), ((32, 1, 2, 8, 16),))) as simd:
            fragment = ShardLayout(
                Layout((2, (2, 4), (2, 2, 2)), (0, (0, 0), (0, 0, 1))), (S(0), S(4), S(2), S(3), S(1)), simd
            )
            single = tf.matmul(tf.reshard(a, fragment, "rmem"), tf.reshard(b, fragment, "rmem"))
            integer = tf.matmul(tf.reshard(ai, fragment, "rmem"), tf.reshard(bi, fragment, "rmem"))
            memory = tf.matmul(a, b)
            return single, integer, memory
