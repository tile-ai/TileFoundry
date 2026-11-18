"""The name h is bound to two values."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Rebound:
    @func
    def forward(x: Tensor[(4, 3), "f32"]):
        h = x * x
        h = h * x
        return h
