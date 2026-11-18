"""An integer parameter beside a floating point one."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class IntParam:
    @func
    def forward(x: Tensor[(4, 3), "f32"], index: Tensor[(2,), "i64"]):
        return tf.index_select(x, index, dim=0) * 2.0
