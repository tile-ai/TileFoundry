"""Matmuls written with default layouts, with ``@``, and with ``tf.matmul``."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="default_layout", target=CpuTarget())
class DefaultLayout:
    @func
    def default_layout(a: Tensor[(2, 3), "f32"], b: Tensor[(3, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.matmul(a, b)


@module(entry="infix", target=CpuTarget())
class InfixMatmul:
    @func
    def infix(a: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]) -> Tensor[(2, 3), "bf16"]:
        return a @ b


@module(entry="spelled", target=CpuTarget())
class SpelledMatmul:
    @func
    def spelled(a: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]) -> Tensor[(2, 3), "bf16"]:
        return tf.matmul(a, b)
