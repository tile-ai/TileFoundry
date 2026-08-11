"""A thread-sharded RMS norm over two 1536-element rows."""

from tilefoundry import module
from tilefoundry.dsl import *


@module(entry="rmsnorm_seq_2", topologies=(Topology("thread", 2 * 4 * 32),))
class RmsnormSeq2Module:
    @func
    def rmsnorm_seq_2(a: Tensor[(2, 1536), "bf16"]):
        with Mesh(("thread",), (2, 4, 32), ("x", "y", "t")) as m:
            a_reg = tf.reshard(a, (2 @ m.x, 12 @ m.y, 128 @ m.t), "rmem")
            a_f32 = tf.cast(a_reg, "f32")
            a_sq = tf.square(a_f32)
            a_mean = tf.reduce(a_sq, (-1,), True, ReduceKind.MEAN)
            a_inv = tf.rsqrt(a_mean + 1e-6)
            a_norm_f32 = a_f32 * a_inv
            a_norm = tf.cast(a_norm_f32, "bf16")
            return tf.reshard(a_norm, (2 @ m.x, 12 @ m.y, 128 @ m.t), "gmem")
