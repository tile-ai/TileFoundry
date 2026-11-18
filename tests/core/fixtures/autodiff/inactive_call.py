"""Calls no gradient reaches: a sibling and a child slot read only a parameter not in wrt."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward")
class Offset:
    @func
    def forward(c: Tensor[(4, 3), "f32"]):
        return c + 1.0


@module(entry="forward", target=cpu.Target())
class InactiveCall:
    offset = Offset

    @func
    def scale(c: Tensor[(4, 3), "f32"]):
        return c * 2.0

    @func
    def forward(x: Tensor[(4, 3), "f32"], c: Tensor[(4, 3), "f32"]):
        k = scale(c)
        b = offset(c)
        return x * k + b
