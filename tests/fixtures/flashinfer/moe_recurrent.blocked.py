"""MoE dispatch into a grouped matmul that has no current HIR surface.

Notes:
upstream: flashinfer-ai/flashinfer @ 2ab910c5 csrc/moe.cu:grouped_gemm
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: runtime_expression: unsupported call 'tf.grouped_matmul' (2 positional, no keywords)
ledger: OP-05
placement_refusal: IndexSelect rejects the sharded smem operand on dim 0.
placement_error: IndexSelect: dim 0 index_select over a shard layout with multiple
Split axes including the selected dim; cannot derive an output layout
placement_workaround: materialize the staged tensor back to gmem before IndexSelect.
"""

from tilefoundry import module
from tilefoundry.dsl import Mesh, Tensor, Topology, func, tf
from tilefoundry.target import CudaTarget


@module(
    entry="grouped",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class MoeRecurrent:
    @func
    def grouped(
        tokens: Tensor[(8, 16), "bf16"],
        weights: Tensor[(8, 16), "bf16"],
        route: Tensor[(4,), "i32"],
    ):
        with Mesh(("cta",), layout=(2,), names=("tile",)) as cta:
            with Mesh(("thread",), layout=(4,), names=("lane",)) as thread:
                tokens_smem = tf.reshard(tokens, (8 @ cta.tile, 16 @ thread.lane), "smem")
                tokens_for_gather = tf.reshard(tokens_smem, (8, 16), "gmem")
                weights_smem = tf.reshard(weights, (8, 16), "smem")
                route_rmem = tf.reshard(route, (4,), "rmem")
                routed = tf.index_select(tokens_for_gather, route_rmem, dim=0)
                routed_smem = tf.reshard(routed, (4 @ cta.tile, 16 @ thread.lane), "smem")
                return tf.grouped_matmul(routed_smem, weights_smem)
