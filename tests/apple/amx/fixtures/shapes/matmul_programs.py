"""AMX matmul programs for analysis and target-boundary checks."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_apple import AmxTarget


@func(target=AmxTarget())
def amx_f32_gemm(
    x: Tensor[(64, 128), "f32"], w: Tensor[(128, 64), "f32"]
) -> Tensor[(64, 64), "f32"]:
    return tf.matmul(x, w)


@func(target=AmxTarget())
def amx_register_sized_f32_gemm(
    x: Tensor[(16, 8), "f32"], w: Tensor[(8, 16), "f32"]
) -> Tensor[(16, 16), "f32"]:
    return tf.matmul(x, w)


@func(target=AmxTarget())
def amx_coarse_m_f32_gemm(
    x: Tensor[(8, 8), "f32"], w: Tensor[(8, 16), "f32"]
) -> Tensor[(8, 16), "f32"]:
    return tf.matmul(x, w)


@func(target=AmxTarget())
def amx_odd_m_f32_gemm(
    x: Tensor[(18, 128), "f32"], w: Tensor[(128, 64), "f32"]
) -> Tensor[(18, 64), "f32"]:
    return tf.matmul(x, w)


@func(target=AmxTarget())
def amx_odd_n_f32_gemm(
    x: Tensor[(64, 128), "f32"], w: Tensor[(128, 18), "f32"]
) -> Tensor[(64, 18), "f32"]:
    return tf.matmul(x, w)


@func(target=AmxTarget())
def amx_bf16_gemm(
    x: Tensor[(64, 128), "bf16"], w: Tensor[(128, 64), "bf16"]
) -> Tensor[(64, 64), "bf16"]:
    return tf.matmul(x, w)
