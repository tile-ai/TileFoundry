"""Dims that only a callee declares, as authored programs.

A staged operator gives its helper a tiling axis the entry's signature never
mentions: the chunk length is an axis of the helper's own scratch, not of any
value crossing the call. ``Staged`` keeps the helper in the same module,
``Rooted`` behind a module boundary.

``DeadScratch`` names the same dimension in a binding nothing reads; the
block drops what it does not use, so no function in that program declares it.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget

BT = DimVar("BT", 0, 1 << 10)


@module(entry="main", target=CpuTarget())
class Staged:
    """One root calling a helper that alone declares the chunk length."""

    @func
    def stage(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        y = tf.zeros(Tensor[(BT,), "f32"])
        s = tf.reduce(y, (0,), False, ReduceKind.SUM)
        return x + s

    @func
    def main(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return stage(x)  # noqa: F821


@module(entry="stage", target=CpuTarget())
class Helper:
    """The same helper standing alone, so a root can attach it."""

    @func
    def stage(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        y = tf.zeros(Tensor[(BT,), "f32"])
        s = tf.reduce(y, (0,), False, ReduceKind.SUM)
        return x + s


@module(entry="main", target=CpuTarget())
class Rooted:
    """One root reaching the helper as an attached child."""

    child = Helper

    @func
    def main(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return child(x)  # noqa: F821


@module(entry="main", target=CpuTarget())
class DeadScratch:
    """A chunk length named only by a binding the block never uses."""

    @func
    def main(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        _scratch = tf.empty(Tensor[(BT,), "f32"])
        return x
