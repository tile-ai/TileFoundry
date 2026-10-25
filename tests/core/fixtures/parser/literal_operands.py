"""A float literal operand and an explicit cast that gives a literal the operand dtype."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="add_literal", target=CpuTarget())
class FloatLiteral:
    @func
    def add_literal(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return x + 1.0


@module(entry="scale", target=CpuTarget())
class CastLiteral:
    @func
    def scale(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return x * tf.cast(-0.5, dtype="bf16")
