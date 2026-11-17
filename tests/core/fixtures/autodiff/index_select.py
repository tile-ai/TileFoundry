"""Rows selected by an index, some selected twice."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class IndexSelect:
    @func
    def forward(table: Tensor[(5, 3), "f32"], index: Tensor[(6,), "i64"]):
        rows = tf.index_select(table, index, dim=0)
        return rows * rows
