"""Concat and stack inputs written as a list, a tuple, and a static comprehension."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="from_list", target=cpu.Target())
class ConcatFromList:
    @func
    def from_list(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.concat([a, b], axis=0)


@module(entry="from_tuple", target=cpu.Target())
class ConcatFromTuple:
    @func
    def from_tuple(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.concat((a, b), axis=0)


@module(entry="stack_rows", target=cpu.Target())
class StackRows:
    @func
    def stack_rows(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.stack([a, b], axis=0)


@module(entry="rows", target=cpu.Target())
class ComprehensionRows:
    @func
    def rows(x: Tensor[(2, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.stack([x[index, :] for index in range(2)], axis=0)
