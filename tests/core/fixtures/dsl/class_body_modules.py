"""Class bodies that make a Module: functions, an orchestration method, methods alone."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="composed", target=CpuTarget())
class _Demo:
    @func
    def leaf(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)

    @func
    def composed(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        y = leaf(x, g)
        return tf.add(y, y)


@module(entry="only", target=CpuTarget())
class _WithMethod:
    @func
    def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)

    def describe(self):  # noqa: ANN001 — orchestration method, bound on the Module
        return f"{self.name}/{self.entry}"


@module(target=CpuTarget())
class _MethodsOnly:
    def walk(self, x):
        return x


@module(target=CpuTarget())
class _NoStep:
    @func
    def helper(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)
