"""A Module whose callee is named backward, the name the generated entry takes."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class BackwardName:
    @func
    def backward(x: Tensor[(4, 3), "f32"]):
        return x * x

    @func
    def forward(x: Tensor[(4, 3), "f32"]):
        return backward(x) * x
