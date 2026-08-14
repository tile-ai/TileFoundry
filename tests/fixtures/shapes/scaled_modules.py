"""Weighted child-module compositions used by analysis, evaluation, and runtime."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, DimVar, Tensor, tf
from tilefoundry.target import CudaTarget

EVALUATOR_N = DimVar("N_eval", 1, 8)
RESOURCE_N = DimVar("n_resource", 1, 9)


@module(entry="run")
class ScaledChild:
    @func
    def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.mul(x, w)


@module(entry="run")
class DynamicScaledChild:
    @func
    def run(
        x: Tensor[(EVALUATOR_N,), "f32"],
        w: ConstTensor[(EVALUATOR_N,), "f32"],
    ) -> Tensor[(EVALUATOR_N,), "f32"]:
        return tf.mul(x, w)


@module(entry="run")
class BroadcastScaledChild:
    @func
    def run(
        x: Tensor[(RESOURCE_N,), "f32"], w: ConstTensor[(1,), "f32"]
    ) -> Tensor[(RESOURCE_N,), "f32"]:
        return tf.mul(x, w)


@module(entry="fused", target=CudaTarget("nvidia.h200_sxm"))
class FusedScaledParent:
    scaled = ScaledChild

    @func
    def fused(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return scaled(x)  # noqa: F821 -- class-body binding


@module(entry="both", target=CudaTarget("nvidia.h200_sxm"))
class PairedScaledParent:
    left = ScaledChild
    right = ScaledChild

    @func
    def both(w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.add(left(w), right(w))  # noqa: F821 -- class-body bindings
