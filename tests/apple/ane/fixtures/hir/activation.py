"""ReLU offered to ANE as a nonlinear applied by a pass-through task.

ANE has no sigmoid, and refuses fp32 operands.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="hir_activation", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def hir_activation(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"],
        xs: Tensor[(1, 32, 16, 16), "f32", ((1, 32, 16, 16), (8192, 256, 16, 1)), "gmem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            rectified = tf.relu(x)
            squashed = tf.sigmoid(x)
            single = tf.relu(xs)
            return rectified, squashed, single
