"""Two different Modules named Block are reachable from one Module."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


def block(scale):
    @module(entry="forward")
    class Block:
        @func
        def forward(x: Tensor[(4, 3), "f32"]):
            h = x * scale
            return h * x

    return Block


@module(entry="forward", target=cpu.Target())
class SameName:
    first = block(2.0)
    second = block(3.0)

    @func
    def forward(x: Tensor[(4, 3), "f32"]):
        return second(first(x))
