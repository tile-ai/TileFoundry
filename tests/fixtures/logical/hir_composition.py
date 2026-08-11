"""Three minimal shapes of one composed kernel, as authored programs.

A root calling its own helper, a root calling an attached child, and a root
reaching both from one entry, exported together as ``REFERENCE_PROGRAMS``.
These are reference programs, not a behaviour matrix: what a composed call
means is asserted where that behaviour lives.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = (Topology("cta", 132),)


@module(entry="run")
class Expert:
    """A child whose weight is its own, so a caller passes activations alone."""

    @func
    def run(
        x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 8), "f32"]
    ) -> Tensor[(4, 8), "f32"]:
        return tf.matmul(x, w)


@module(entry="root", target=_H200, topologies=_CTA)
class SameModule:
    """One root calling a helper Function of its own Module."""

    @func
    def scale(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return tf.mul(x, x)

    @func
    def root(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return scale(x)  # noqa: F821


@module(entry="root", target=_H200, topologies=_CTA)
class CrossModule:
    """One root calling the entry of an attached child Module."""

    expert = Expert

    @func
    def root(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return expert(x)  # noqa: F821


@module(entry="root", target=_H200, topologies=_CTA)
class Fused:
    """One root reaching a sibling Function and an attached child."""

    expert = Expert

    @func
    def scale(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return tf.mul(x, x)

    @func
    def root(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return tf.add(scale(x), expert(x))  # noqa: F821


REFERENCE_PROGRAMS = (
    ("same_module", SameModule),
    ("cross_module", CrossModule),
    ("fused", Fused),
)
