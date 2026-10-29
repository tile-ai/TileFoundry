"""Pointwise HIR inputs are authored independently of AMX operand Patterns."""
from tilefoundry.dsl import *
from tilefoundry_apple import AmxTarget

@module(entry="hir_elementwise", target=AmxTarget(), topologies=(Topology("thread", 12),))
class Program:
    @func
    def hir_elementwise(
        h: Tensor[(1, 32), "f16", ((1, 32), (0, 1)), "rmem"],
        h2: Tensor[(1, 32), "f16", ((1, 32), (0, 1)), "rmem"],
        s: Tensor[(1, 16), "f32", ((1, 16), (0, 1)), "rmem"],
        s2: Tensor[(1, 16), "f32", ((1, 16), (0, 1)), "rmem"],
        d: Tensor[(1, 8), "f64", ((1, 8), (0, 1)), "rmem"],
        d2: Tensor[(1, 8), "f64", ((1, 8), (0, 1)), "rmem"],
        i: Tensor[(1, 32), "i16", ((1, 32), (0, 1)), "rmem"],
        i2: Tensor[(1, 32), "i16", ((1, 32), (0, 1)), "rmem"],
        u: Tensor[(1, 32), "u16", ((1, 32), (0, 1)), "rmem"],
    ):
        with Mesh(("thread",), layout=(4,), names=("worker",)) as _threads:
            half_mul = tf.mul(h, h2)
            single_mul = tf.mul(s, s2)
            double_mul = tf.mul(d, d2)
            integer_mul = tf.mul(i, i2)
            single_add = tf.add(s, s2)
            single_sub = tf.sub(s, s2)
            integer_add = tf.add(i, i2)
            integer_sub = tf.sub(i, i2)
            single_min = tf.min(s, s2)
            single_max = tf.max(s, s2)
            half_neg = tf.neg(h)
            single_neg = tf.neg(s)
            double_neg = tf.neg(d)
            unsigned_mul = tf.mul(u, u)
            return half_mul, single_mul, double_mul, integer_mul, single_add, single_sub, integer_add, integer_sub, single_min, single_max, half_neg, single_neg, double_neg, unsigned_mul
