"""Signed literals, which are constants, and a negated value, which is a call."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="negative_float_literal", target=CpuTarget())
class NegativeFloatLiteral:
    @func
    def negative_float_literal(x: Tensor[(), "f32"]):
        return -1.0


@module(entry="negative_integer_literal", target=CpuTarget())
class NegativeIntegerLiteral:
    @func
    def negative_integer_literal(x: Tensor[(), "f32"]):
        return -3


@module(entry="negative_infinity_literal", target=CpuTarget())
class NegativeInfinityLiteral:
    @func
    def negative_infinity_literal(x: Tensor[(), "f32"]):
        return -1e999


@module(entry="negated_value", target=CpuTarget())
class NegatedValue:
    @func
    def negated_value(x: Tensor[(), "f32"]):
        return -x
