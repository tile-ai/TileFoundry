"""A function that calls the bare ``add`` alias, captured from ``tilefoundry.dsl.tf``."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.dsl import tf
from tilefoundry.target import cpu


@module(entry="_alias_call", target=cpu.Target())
class AliasCall:
    @func
    def _alias_call(
        a: Tensor[(8,), DType.f32],
        b: Tensor[(8,), DType.f32],
    ) -> Tensor[(8,), DType.f32]:

        return tf.add(a, b)
