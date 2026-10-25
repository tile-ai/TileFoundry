"""Concat and stack inputs written as a list, a tuple, and a static comprehension."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="from_list", target=CpuTarget())
class ConcatFromList:
    @func
    def from_list(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.concat([a, b], axis=0)


@module(entry="from_tuple", target=CpuTarget())
class ConcatFromTuple:
    @func
    def from_tuple(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.concat((a, b), axis=0)


@module(entry="stack_rows", target=CpuTarget())
class StackRows:
    @func
    def stack_rows(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.stack([a, b], axis=0)


@module(entry="rows", target=CpuTarget())
class ComprehensionRows:
    @func
    def rows(x: Tensor[(2, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.stack([x[index, :] for index in range(2)], axis=0)
