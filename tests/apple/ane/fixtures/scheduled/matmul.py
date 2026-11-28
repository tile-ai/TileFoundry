"""A bypass relays a row-major left operand channel-outer in the on-chip buffer.

A 1x1 convolution then multiplies it by the row-major right operand.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="matmul", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def matmul(
        a: Tensor[(8, 16), "f16", ((8, 16), (16, 1)), "gmem"],
        b: Tensor[(16, 16), "f16", ((16, 16), (16, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            al = tf.schedule((a, tf.empty(Tensor[(8, 16), "f16", ((8, 16), (16, 1)), "smem"])), op=T.ane.load())
            outer = tf.schedule((al, tf.empty(Tensor[(8, 16), "f16", ((8, 16), (1, 8)), "smem"])), op=T.ane.bypass(perm=(2, 1, 0)))
            y = tf.schedule((outer, b, tf.empty(Tensor[(8, 16), "f16", ((8, 16), (16, 1)), "smem"])), op=T.ane.conv())
            return tf.schedule((y, tf.empty(Tensor[(8, 16), "f16", ((8, 16), (16, 1)), "gmem"])), op=T.ane.store())
