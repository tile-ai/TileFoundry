"""A bf16 tensor added to an integer literal, which is i64."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="add_integer", target=CpuTarget())
class Bf16IntegerOperand:
    @func
    def add_integer(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return x + 1
