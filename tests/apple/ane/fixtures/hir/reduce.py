"""Sum, maximum and minimum reductions offered to ANE over the spatial axes or the width.

ANE refuses a mean, and a reduction over the channel axis or the height alone.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="hir_reduce", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def hir_reduce(
        x: Tensor[(1, 32, 16, 16), "f16", ((1, 32, 16, 16), (8192, 256, 16, 1)), "smem"],
    ):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            total = tf.reduce(x, axes=(2, 3), keepdim=True, kind=ReduceKind.SUM)
            largest = tf.reduce(x, axes=(2, 3), keepdim=True, kind=ReduceKind.MAX)
            smallest = tf.reduce(x, axes=(2, 3), keepdim=True, kind=ReduceKind.MIN)
            mean = tf.reduce(x, axes=(2, 3), keepdim=True, kind=ReduceKind.MEAN)
            width = tf.reduce(x, axes=(3,), keepdim=True, kind=ReduceKind.SUM)
            flat = tf.reduce(x, axes=(2, 3), keepdim=False, kind=ReduceKind.SUM)
            channels = tf.reduce(x, axes=(1,), keepdim=True, kind=ReduceKind.SUM)
            height = tf.reduce(x, axes=(2,), keepdim=True, kind=ReduceKind.SUM)
            return total, largest, smallest, mean, width, flat, channels, height
