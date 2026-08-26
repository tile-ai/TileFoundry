"""Paged KV gather whose base index selection reports the wrong traffic.

Notes:
upstream: flashinfer-ai/flashinfer@2ab910c5 page.cu:gather_paged_kv
license: Apache-2.0
blocked: mis-analyzed
phase: selection/analysis
got: traffic traffic=gmem:r136/w32@r136/w32
expected: only the selected page bytes
why: the access relation charges the whole cache instead of selected pages
ledger: OP-04
placement_refusal: IndexSelect rejects the sharded smem operand on dim 0.
placement_error: IndexSelect: dim 0 index_select over a shard layout with multiple
Split axes including the selected dim; cannot derive an output layout
placement_workaround: materialize the staged tensor back to gmem before IndexSelect.
"""

from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CudaTarget


@module(
    entry="paged_gather",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class PagedGather:
    @func
    def paged_gather(cache: Tensor[(8, 4), "f32"], indices: Tensor[(2,), "i32"]):
        with Mesh(("cta",), layout=(2,), names=("tile",)) as cta:
            with Mesh(("thread",), layout=(4,), names=("lane",)) as thread:
                cache_rmem = tf.reshard(cache, (2 @ cta.tile, 1, 4 @ thread.lane), "rmem")
                cache_smem = tf.reshard(cache_rmem, (2 @ cta.tile, 1, 4 @ thread.lane), "smem")
                cache_for_gather = tf.reshard(cache_smem, (8, 4), "gmem")
                return tf.index_select(cache_for_gather, indices, dim=0)
