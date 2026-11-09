"""Class bodies that make a Module: functions, an orchestration method, methods alone."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="composed", target=cpu.Target())
class _Demo:
    @func
    def leaf(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)

    @func
    def composed(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        y = leaf(x, g)
        return tf.add(y, y)


@module(entry="only", target=cpu.Target())
class _WithMethod:
    @func
    def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)

    def describe(self):
        return f"{self.name}/{self.entry}"


@module(target=cpu.Target())
class _MethodsOnly:
    def walk(self, x):
        return x


@module(target=cpu.Target())
class _NoStep:
    @func
    def helper(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)
