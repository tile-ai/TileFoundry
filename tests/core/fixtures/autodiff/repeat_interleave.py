"""Each element repeated along an inner axis."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class RepeatInterleave:
    @func
    def forward(x: Tensor[(2, 3, 4), "f32"]):
        repeated = tf.repeat_interleave(x, repeats=2, axis=1)
        return repeated * repeated
