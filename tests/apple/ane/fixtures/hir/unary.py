"""Absolute value, square and negation offered to ANE as one affine task with options.

ANE has no rounding.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="hir_unary", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def hir_unary(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (16384, 512, 32, 1)), "smem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            absolute = tf.abs(x)
            squared = tf.square(x)
            negated = tf.neg(x)
            rounded = tf.round(x)
            return absolute, squared, negated, rounded
