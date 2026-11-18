"""A callee that returns the empty tuple: its forward returns only its saved tuple."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class VoidCallee:
    @func
    def nothing(x: Tensor[(4,), "f32"]):
        return ()

    @func
    def forward(x: Tensor[(4,), "f32"]):
        n = nothing(x)
        return x * x, n
