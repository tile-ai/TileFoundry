"""IndexSelect with a fill_value for indices out of range."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class FilledIndexSelect:
    @func
    def forward(table: Tensor[(5, 3), "f32"], index: Tensor[(6,), "i64"]):
        return tf.index_select(table, index, dim=0, fill_value=0.0) * 2.0
