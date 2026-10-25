"""A bf16 tensor added to a float literal, which is f32."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="add_literal", target=CpuTarget())
class Bf16FloatLiteral:
    @func
    def add_literal(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return x + 1.0
