"""Matmul programs spanning analysis, CUDA, and AMX target boundaries."""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor, tf
from tilefoundry.target import AmxTarget, CudaTarget

DYNAMIC_M = DimVar("seq", 1, 128)


@func
def gemm_rms_norm(
    x: Tensor[(2, 4), "f32"], w: Tensor[(4, 2), "f32"], weight: Tensor[(2,), "f32"]
) -> Tensor[(2, 2), "f32"]:
    h = tf.matmul(x, w)
    return tf.rms_norm(h, weight)


@func(target=CudaTarget("nvidia.h200_sxm"))
def bf16_gemm_rms_norm(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "bf16"]:
    h = tf.matmul(x, w)
    return tf.rms_norm(h, weight)


@func(target=CudaTarget("nvidia.h200_sxm"))
def cuda_bf16_gemm(
    x: Tensor[(64, 128), "bf16"], w: Tensor[(128, 64), "bf16"]
) -> Tensor[(64, 64), "bf16"]:
    return tf.matmul(x, w)


@func(target=CudaTarget("nvidia.h200_sxm"))
def cuda_f32_gemm(
    x: Tensor[(64, 128), "f32"], w: Tensor[(128, 64), "f32"]
) -> Tensor[(64, 64), "f32"]:
    return tf.matmul(x, w)


@func(target=CudaTarget("nvidia.h200_sxm"))
def cuda_odd_m_bf16_gemm(
    x: Tensor[(15, 128), "bf16"], w: Tensor[(128, 64), "bf16"]
) -> Tensor[(15, 64), "bf16"]:
    return tf.matmul(x, w)


@func
def dynamic_bf16_gemm(
    x: Tensor[(DYNAMIC_M, 4), "bf16"], w: Tensor[(4, 2), "bf16"]
) -> Tensor[(DYNAMIC_M, 2), "bf16"]:
    h = tf.matmul(x, w)
    return h


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
