"""Lifecycle and packed lookup are explicit TIR, with no invented HIR sites."""
from tilefoundry.dsl import *
from tilefoundry_apple import AmxTarget
from tilefoundry_apple.amx.ir.tir.common import ExtractMode, LutMode


@module(entry="amx_explicit", target=AmxTarget(), topologies=(Topology("thread", 12),))
class Program:
    @func
    def amx_explicit(
        token: Tensor[(1,), "u8", ((1,), (1,)), "rmem"],
        indices: Tensor[(4,), "u8", ((4,), (1,)), "rmem"],
        table: Tensor[(16,), "f32", ((16,), (1,)), "rmem"],
        copy_source: Tensor[(1, 16), "f32", ((1, 16), (0, 1)), "rmem"],
        extracted: Tensor[(1, 16), "f32", ((1, 16), (0, 1)), "rmem"],
        destination: Tensor[(16,), "f32", ((16,), (1,)), "rmem"],
    ):
        with Mesh(("thread",), layout=(12,), names=("worker",)) as _threads:
            enabled = tf.schedule((token,), op=T.amx.set())
            lookup = tf.schedule((indices, table, destination), op=T.amx.genlut_v2(mode=LutMode.LOOKUP32_2, dtype=DType.f32))
            extracted_x = tf.schedule((copy_source, extracted), op=T.amx.extrx_v2(mode=ExtractMode.COPY, src_dtype=DType.f32, dst_dtype=DType.f32))
            extracted_y = tf.schedule((copy_source, extracted_x), op=T.amx.extry_v2(mode=ExtractMode.COPY, src_dtype=DType.f32, dst_dtype=DType.f32))
            disabled = tf.schedule((enabled,), op=T.amx.clr())
            return lookup, extracted_y, disabled
