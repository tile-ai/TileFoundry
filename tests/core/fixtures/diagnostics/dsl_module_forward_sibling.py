"""A Module function that calls a sibling defined below it."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="caller", target=CpuTarget())
class _Forward:
    @func
    def caller(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        y = callee(x, g)  # noqa: F821 — defined below, unresolved on purpose
        return tf.add(y, y)

    @func
    def callee(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)
