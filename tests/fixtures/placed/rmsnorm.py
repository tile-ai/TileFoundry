"""A thread-sharded RMS norm over one 1536-element row."""

from tilefoundry import module
from tilefoundry.dsl import *


@module(entry="rmsnorm", topologies=(Topology("thread", 6 * 32),))
class RmsnormModule:
    @func
    def rmsnorm(a: Tensor[(1, 1536), "bf16"]):
        with Mesh(("thread",), (6, 32), ("w", "t")) as m:
            a_reg = tf.reshard(a, (1, 1536 @ (m.w, m.t)), "rmem")
            a_f32 = tf.cast(a_reg, "f32")
            a_sq = tf.square(a_f32)
            a_mean = tf.reduce(a_sq, (-1,), True, ReduceKind.MEAN)
            a_inv = tf.rsqrt(a_mean + 1e-6)
            a_norm_f32 = a_f32 * a_inv
            a_norm = tf.cast(a_norm_f32, "bf16")

            return tf.reshard(a_norm, (1, 1536 @ (m.w, m.t)), "gmem")
