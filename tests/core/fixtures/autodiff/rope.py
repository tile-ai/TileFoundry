"""RoPE as the models call it: one input passed twice, one result kept; halves repeat."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Rope:
    @func
    def forward(
        q: Tensor[(1, 4, 2, 6), "f32"], k: Tensor[(1, 4, 2, 6), "f32"], half_cos: Tensor[(8, 3), "f32"],
        half_sin: Tensor[(8, 3), "f32"], pos: Tensor[(4,), "i32"],
    ):
        cos = tf.concat([half_cos, half_cos], axis=-1)
        sin = tf.concat([half_sin, half_sin], axis=-1)
        rq, _ = tf.rope(q, q, cos, sin, pos)
        both = tf.rope(rq, k, cos, sin, pos)
        return both[0] * both[1]
