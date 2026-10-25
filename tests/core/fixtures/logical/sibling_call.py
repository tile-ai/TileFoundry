"""A callee declared above its caller.

This is what the printer emits for any multi-function Module.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="caller", target=CpuTarget())
class _Siblings:
    @func
    def callee(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @func
    def caller(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return callee(x)
