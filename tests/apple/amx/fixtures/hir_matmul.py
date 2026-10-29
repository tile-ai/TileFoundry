"""Independent register-resident HIR matrix products; no accumulator is seeded here.

Candidates describe instructions requiring a later explicit zero/accumulate schedule.
K=2 and multiple N fragments exercise repetition rather than a single outer product.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import AmxTarget


@module(entry="hir_matmul", target=AmxTarget(), topologies=(Topology("amx", 1),))
class Program:
    @func
    def hir_matmul(
        h: Tensor[(32, 2), "f16", ((32, 2), (1, 32)), "rmem"],
        ht: Tensor[(2, 64), "f16", ((2, 64), (64, 1)), "rmem"],
        s: Tensor[(16, 2), "f32", ((16, 2), (1, 16)), "rmem"],
        st: Tensor[(2, 64), "f32", ((2, 64), (64, 1)), "rmem"],
        d: Tensor[(8, 2), "f64", ((8, 2), (1, 8)), "rmem"],
        dt: Tensor[(2, 64), "f64", ((2, 64), (64, 1)), "rmem"],
        i: Tensor[(32, 2), "i16", ((32, 2), (1, 32)), "rmem"],
        it: Tensor[(2, 64), "i16", ((2, 64), (64, 1)), "rmem"],
        odd: Tensor[(17, 2), "f32", ((17, 2), (1, 17)), "rmem"],
        square: Tensor[(16, 16), "f32", ((16, 16), (1, 16)), "rmem"],
        square_t: Tensor[(16, 64), "f32", ((16, 64), (64, 1)), "rmem"],
        memory: Tensor[(16, 2), "f32"],
        memory_t: Tensor[(2, 16), "f32"],
    ):
        with Mesh(("amx",), layout=(1,), names=("unit",)) as _amx:
            half = tf.matmul(h, ht)
            single = tf.matmul(s, st)
            double = tf.matmul(d, dt)
            integer = tf.matmul(i, it)
            integer_wide = tf.matmul(i, it, out_dtype="i32")
            odd_m = tf.matmul(odd, st)
            transposed_axes = tf.matmul(square, square_t, a_layout="KM")
            ordinary_memory = tf.matmul(memory, memory_t)
            return half, single, double, integer, integer_wide, odd_m, ordinary_memory, transposed_axes
