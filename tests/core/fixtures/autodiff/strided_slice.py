"""A Slice that takes every second row, which InsertSlice cannot write back."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class StridedSlice:
    @func
    def forward(x: Tensor[(4, 3), "f32"]):
        return x[0:4:2, :] * 2.0
