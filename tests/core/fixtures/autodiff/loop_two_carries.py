"""Two carries that read each other, in a range loop that starts at 1 with step 2."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class LoopTwoCarries:
    @func
    def forward(a0: Tensor[(2, 3), "f32"], b0: Tensor[(2, 3), "f32"], w: Tensor[(2, 3), "f32"]):
        a = a0
        b = b0
        for _ in range(1, 7, 2):
            a = a * b + w
            b = b * tf.sigmoid(a)
        return a * b
