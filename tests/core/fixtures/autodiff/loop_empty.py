"""A loop that runs no iteration: its carry passes through."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class LoopEmpty:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(4, 3), "f32"]):
        h = x * x
        for _ in range(3, 3):
            h = h * w
        return h
