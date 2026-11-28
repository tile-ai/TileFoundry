"""Permutations of the channel, height and width axes offered to ANE as a pass-through task.

ANE refuses a permutation of the batch axis.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="hir_transpose", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def hir_transpose(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            swapped = tf.transpose(x, perm=(0, 3, 2, 1))
            rotated = tf.transpose(x, perm=(0, 2, 3, 1))
            batch = tf.transpose(x, perm=(1, 0, 2, 3))
            return swapped, rotated, batch
