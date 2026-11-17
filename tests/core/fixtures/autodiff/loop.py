"""A loop whose carry depends on the parameter."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Loop:
    @func
    def forward(x: Tensor[(4, 3), "f32"]):
        h = x
        for _ in range(3):
            h = h * x
        return h
