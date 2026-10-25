"""A function with a declared weight, alone and as a child two owners share."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="scale", target=CpuTarget())
class _Weighted:
    @func
    def scale(x: Tensor[(2,), "f32"], w: ConstTensor[(2,), "f32"]):
        return x * w


@module(entry="scale")
class _Leaf:
    @func
    def scale(x: Tensor[(2,), "f32"], w: ConstTensor[(2,), "f32"]):
        return x * w


@module(entry="twice", target=CpuTarget())
class _Owner:
    leaf = _Leaf

    @func
    def twice(x: Tensor[(2,), "f32"]):
        return x + x
