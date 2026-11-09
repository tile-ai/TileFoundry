"""A call to an op the tf namespace does not have."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="missing", target=cpu.Target())
class UnknownOp:
    @func
    def missing(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return tf.no_such_op(x, x)
