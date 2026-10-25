"""Programs whose work sits on either side of the local pricing boundary."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="unpriced_only",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class _PricingBoundary:
    @func
    def unpriced_only(source: Tensor[(1,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
            local = tf.reshard(source, (1 @ cta.tile,), "rmem")
            moved = tf.reshard(local, (1 @ cta.tile,), "rmem")
            return tf.add(moved, moved)

    @func
    def mixed(source: Tensor[(64,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
            placed = tf.reshard(source, (64 @ cta.tile,), "gmem")
            local = tf.reshard(placed, (64 @ cta.tile,), "rmem")
            return tf.add(local, local)

    @func
    def serviced_predicate(source: Tensor[(64,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
            placed = tf.reshard(source, (64 @ cta.tile,), "gmem")
            return placed <= placed
