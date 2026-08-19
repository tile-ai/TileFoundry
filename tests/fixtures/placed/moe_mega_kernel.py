"""Two same-Module experts on disjoint slices of one CTA topology."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = (Topology("cta", 132),)


@module(entry="experts", target=_H200, topologies=_CTA)
class MoEMegaKernel:
    """Two same-Module experts on disjoint slices of one CTA topology.

    Each branch names its slice with a nested ``Mesh`` scope and reshards into
    it, so the slice reaches the branch primitive's result type rather than
    staying a lexical scope nothing recorded. Each reshards back to the whole
    topology outside its slice, so the two values the join consumes are placed
    the same way and their combination means something.
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
        with Mesh(("cta",), layout=(132,), names=("tile",)) as _whole:
            return tf.add(routed_expert(tokens), shared_expert(tokens))  # noqa: F821
