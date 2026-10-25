"""A function that calls the bare ``add`` alias, captured from ``tilefoundry.dsl.tf``."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.dsl.tf import add as _tf_add  # noqa: F401  -- closure capture
from tilefoundry.target import CpuTarget


@module(entry="_alias_call", target=CpuTarget())
class AliasCall:
    @func
    def _alias_call(
        a: Tensor[(8,), DType.f32],
        b: Tensor[(8,), DType.f32],
    ) -> Tensor[(8,), DType.f32]:

        return _tf_add(a, b)
