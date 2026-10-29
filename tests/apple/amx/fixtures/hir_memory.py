"""Memory transfers and exact HIR cast/reshard boundaries.

A cast keeps its logical shape. It must not silently discard the extra Z
lanes consumed by EXTRX/EXTRY conversion modes.
"""
from tilefoundry.dsl import *
from tilefoundry_apple import AmxTarget


@module(entry="hir_memory", target=AmxTarget(), topologies=(Topology("thread", 12),))
class Program:
    @func
    def hir_memory(
        memory: Tensor[(1, 16), "f32", ((1, 16), (16, 1)), "gmem"],
        registers: Tensor[(1, 16), "f32", ((1, 16), (16, 1)), "rmem"],
        interleaved: Tensor[(1, 16), "f32", Layout((1, (2, 8)), (0, (16, 1))), "rmem"],
        bad_stride: Tensor[(1, 16), "f32", ((1, 16), (0, 2)), "gmem"],
        extract_float: Tensor[(1, 32), "f32", Layout((1, (16, 2)), (1024, (1, 16))), "rmem"],
        extract_int: Tensor[(1, 32), "i32", Layout((1, (16, 2)), (1024, (1, 16))), "rmem"],
        extract_vertical: Tensor[(1, 32), "f32", Layout((1, (16, 2)), (16, (64, 16))), "rmem"],
        z_float: Tensor[(1, 4, 16), "f32", ((1, 4, 16), (1024, 16, 1)), "rmem"],
        z_int: Tensor[(1, 4, 16), "i32", ((1, 4, 16), (1024, 16, 1)), "rmem"],
    ):
        with Mesh(("thread",), layout=(2,), names=("worker",)) as _threads:
            load = tf.reshard(memory, ((1, 16), (16, 1)), "rmem")
            load_interleaved = tf.reshard(memory, Layout((1, (2, 8)), (0, (16, 1))), "rmem")
            store = tf.reshard(registers, ((1, 16), (16, 1)), "gmem")
            store_interleaved = tf.reshard(interleaved, ((1, 16), (16, 1)), "gmem")
            bad_load = tf.reshard(bad_stride, ((1, 16), (0, 1)), "rmem")
            bad_store = tf.reshard(registers, ((1, 16), (0, 2)), "gmem")
            float_extract = tf.cast(extract_float, "f16")
            integer_extract = tf.cast(extract_int, "i16")
            vertical_extract = tf.cast(extract_vertical, "f16")
            identity_cast = tf.cast(registers, "f32")
            float_cast = tf.cast(z_float, "f16")
            integer_cast = tf.cast(z_int, "i16")
            return load, load_interleaved, store, store_interleaved, bad_load, bad_store, float_extract, integer_extract, vertical_extract, identity_cast, float_cast, integer_cast
