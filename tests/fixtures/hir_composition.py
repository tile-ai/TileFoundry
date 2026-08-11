"""Composed HIR roots, as authored programs other work can import.

Three minimal shapes of one kernel -- a root calling its own helper, a root
calling an attached child, and a root reaching both from one entry -- plus the
placed routed/shared program, whose two same-Module experts run on disjoint
slices of one CTA topology and reshard so those placements reach their results.

These are reference programs, not a behaviour matrix: what a composed call means
is asserted where that behaviour lives.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, tf
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


@module(entry="experts", target=_H200, topologies=_CTA)
class MoEMegaKernel:
    """Two same-Module experts on disjoint slices of one CTA topology.

    Each branch names its slice with a nested ``Mesh`` scope and reshards into
    it, so the slice reaches the branch primitive's result type rather than
    staying a lexical scope nothing recorded. Each reshards back outside its
    scope, so the two results meet unplaced.
    """

    @func
    def routed_expert(tokens: Tensor[(120, 64), "f32"]):
        with Mesh(("cta",), layout=(132,), names=("tile",)) as cta:
            with cta[:120] as routed:
                local = tf.reshard(tokens, (120 @ routed.tile, 64), "gmem")
                placed = tf.relu(local)
            return tf.reshard(placed, (120, 64), "gmem")  # noqa: F821

    @func
    def shared_expert(tokens: Tensor[(120, 64), "f32"]):
        with Mesh(("cta",), layout=(132,), names=("tile",)) as cta:
            with cta[120:] as shared:
                local = tf.reshard(tokens, (120 @ shared.tile, 64), "gmem")
                placed = tf.square(local)
            return tf.reshard(placed, (120, 64), "gmem")  # noqa: F821

    @func
    def experts(tokens: Tensor[(120, 64), "f32"]):
        return tf.add(routed_expert(tokens), shared_expert(tokens))  # noqa: F821
