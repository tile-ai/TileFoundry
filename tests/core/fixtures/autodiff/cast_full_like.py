"""A cast to f16 and back, and a constant shaped like the input."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class CastFullLike:
    @func
    def forward(x: Tensor[(4, 3), "f32"]):
        half = tf.cast(x * x, dtype="f16")
        return tf.cast(half, dtype="f32") + tf.full_like(x, value=2.0) * x
