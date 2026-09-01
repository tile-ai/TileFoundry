"""Direct paged gather placement that IndexSelect currently rejects.

Notes:
upstream: flashinfer-ai/flashinfer @ 2ab910c58fdd2392914ea05e2a8714946ac0eef6
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: IndexSelect: dim 0 index_select over a shard layout
ledger: EXT-01a
placement_refusal: IndexSelect rejects the sharded smem operand on dim 0.
placement_workaround: materialize the staged tensor back to gmem before IndexSelect.
"""

from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CudaTarget


@module(
    entry="paged_gather_placement",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class PagedGatherPlacement:
    @func
    def paged_gather_placement(cache: Tensor[(8, 4), "f32"], indices: Tensor[(2,), "i32"]):
        with Mesh(("cta",), layout=(2,), names=("tile",)) as cta:
            with Mesh(("thread",), layout=(4,), names=("lane",)) as thread:
                cache_smem = tf.reshard(cache, (2 @ cta.tile, 1, 4 @ thread.lane), "smem")
                return tf.index_select(cache_smem, indices, dim=0)
