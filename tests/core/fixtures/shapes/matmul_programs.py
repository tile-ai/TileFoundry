"""Target-neutral matmul programs for analysis."""

from __future__ import annotations

from tilefoundry.dsl import *

DYNAMIC_M = DimVar("seq", 1, 127)


@func
def gemm_rms_norm(
    x: Tensor[(2, 4), "f32"], w: Tensor[(4, 2), "f32"], weight: Tensor[(2,), "f32"]
) -> Tensor[(2, 2), "f32"]:
    h = tf.matmul(x, w)
    return tf.rms_norm(h, weight)


@func
def dynamic_bf16_gemm(
    x: Tensor[(DYNAMIC_M, 4), "bf16"], w: Tensor[(4, 2), "bf16"]
) -> Tensor[(DYNAMIC_M, 2), "bf16"]:
    h = tf.matmul(x, w)
    return h
