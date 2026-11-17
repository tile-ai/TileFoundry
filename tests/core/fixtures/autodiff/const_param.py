"""A ConstTensor weight: a constant, so it has no gradient."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class ConstParam:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: ConstTensor[(3, 3), "f32"]):
        return tf.matmul(x, w)
