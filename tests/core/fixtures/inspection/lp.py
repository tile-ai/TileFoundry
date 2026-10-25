from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="lp", target=CpuTarget())
class Lp:
    @func
    def lp(a: Tensor[(4,), "fp8e4m3"], b: Tensor[(4,), "f8e8m0"], c: Tensor[(4,), "f4e2m1"]):
        return (a, b, c)
