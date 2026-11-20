"""A saved value is a slice of a parameter: it keeps the parameter, which is held anyway."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class ParamView:
    @func
    def forward(x: Tensor[(8, 16), "f32"], w: Tensor[(16, 48), "f32"]):
        head = tf.slice(w, (0, 0), sizes=(16, 16), strides=(1, 1))
        h = tf.matmul(x, head)
        return h * h
