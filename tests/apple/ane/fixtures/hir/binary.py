"""Element-wise add, subtract, multiply, minimum and maximum offered to ANE.

ANE refuses division, a broadcast operand and fp32 operands.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="hir_binary", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def hir_binary(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"],
        y: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"],
        c: Tensor[(1, 32, 1, 1), "f16", ((1, 32, 1, 1), (32, 1, 1, 1)), "smem"],
        xs: Tensor[(1, 32, 16, 16), "f32", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"],
        ys: Tensor[(1, 32, 16, 16), "f32", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            added = tf.add(x, y)
            subtracted = tf.sub(x, y)
            multiplied = tf.mul(x, y)
            smaller = tf.min(x, y)
            larger = tf.max(x, y)
            divided = tf.div(x, y)
            broadcast = tf.add(x, c)
            single = tf.add(xs, ys)
            return added, subtracted, multiplied, smaller, larger, divided, broadcast, single
