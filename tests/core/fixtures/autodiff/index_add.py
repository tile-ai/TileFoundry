"""Rows added into a destination by an index, some rows named more than once."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class IndexAdd:
    @func
    def forward(dst: Tensor[(5, 3), "f32"], index: Tensor[(4,), "i64"], src: Tensor[(4, 3), "f32"]):
        out = tf.index_add(dst, index, src, dim=0)
        return out * out
