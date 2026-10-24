"""CUDA matmul programs."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


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
