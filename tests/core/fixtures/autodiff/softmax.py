"""SoftMax along an axis that is not the last."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Softmax:
    @func
    def forward(x: Tensor[(4, 5, 3), "f32"]):
        return tf.softmax(x, axis=1) * x
