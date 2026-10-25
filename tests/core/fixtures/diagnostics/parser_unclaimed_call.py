"""A call to a name nothing defines."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="refused", target=CpuTarget())
class UnclaimedCall:
    @func
    def refused(a: Tensor[(2, 4), "bf16"]) -> Tensor[(2, 4), "bf16"]:
        return no_such_helper(a)  # noqa: F821
