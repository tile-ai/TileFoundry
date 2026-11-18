"""A tuple parameter annotated with the lowercase ``tuple[...]``, which is not a DSL type."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="rejected", target=cpu.Target())
class LowercaseTupleAnnotation:
    @func
    def rejected(pair: tuple[Tensor[(4,), "f32"], Tensor[(4,), "f32"]]) -> Tensor[(4,), "f32"]:
        return pair[0]
