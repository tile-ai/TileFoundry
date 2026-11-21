"""Matrix products offered to ANE as a 1x1 convolution of a channel-outer left operand.

ANE refuses a row-major left operand and fp32 operands.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="hir_matmul", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def hir_matmul(
        a: Tensor[(256, 2048), "f16", ((256, 2048), (1, 256)), "gmem"],
        ar: Tensor[(256, 2048), "f16", ((256, 2048), (2048, 1)), "gmem"],
        b: Tensor[(2048, 2048), "f16", ((2048, 2048), (2048, 1)), "gmem"],
        a32: Tensor[(256, 2048), "f32", ((256, 2048), (1, 256)), "gmem"],
        b32: Tensor[(2048, 2048), "f32", ((2048, 2048), (2048, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            product = tf.matmul(a, b)
            rows = tf.matmul(ar, b)
            single = tf.matmul(a32, b32)
            return product, rows, single
