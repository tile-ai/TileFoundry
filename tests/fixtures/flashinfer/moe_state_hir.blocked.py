"""Executable-shaped expected HIR for representative MoE/state mechanisms.

Notes:
upstream: flashinfer-ai/flashinfer @ 2ab910c58fdd2392914ea05e2a8714946ac0eef6
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: runtime_expression: unsupported call 'tf.route_topk' (1 positional, keywords ['experts', 'k'])
ledger: OP-05, OP-06, OP-11, OP-13
Toy extents and proposed operations retain the complete mechanism inventory.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

TARGET = CudaTarget("nvidia.h200_sxm")
CUDA_TOPOS = (Topology("cta", 8), Topology("thread", 128))
TOKENS, HIDDEN, EXPERTS, TOPK = 128, 64, 16, 2


# noqa
@module(entry="fused_cta", target=TARGET, topologies=CUDA_TOPOS)
class M04FullMoE:
    @func
    def unfused(
        x: Tensor[(TOKENS, HIDDEN), "bf16"],
        w1: Tensor[(EXPERTS, HIDDEN, 2 * HIDDEN), "bf16"],
        w2: Tensor[(EXPERTS, HIDDEN, HIDDEN), "bf16"],
    ) -> Tensor[(TOKENS, HIDDEN), "bf16"]:
        weights, experts = tf.route_topk(x, experts=EXPERTS, k=TOPK)
        permuted = tf.expert_gather(x, experts)
        gate_up = tf.grouped_matmul(permuted, w1)
        hidden = tf.silu_and_mul(gate_up)
        expert_out = tf.grouped_matmul(hidden, w2)
        return tf.weighted_scatter_reduce(expert_out, weights, experts)

    @func
    def fused_program(
        x: Tensor[(TOKENS, HIDDEN), "bf16"],
        w1: Tensor[(EXPERTS, HIDDEN, 2 * HIDDEN), "bf16"],
        w2: Tensor[(EXPERTS, HIDDEN, HIDDEN), "bf16"],
    ) -> Tensor[(TOKENS, HIDDEN), "bf16"]:
        return tf.fused_moe(x, w1, w2, experts=EXPERTS, k=TOPK)

    @func
    def fused_cta(
        x: Tensor[(TOKENS, HIDDEN), "bf16"],
        w1: Tensor[(EXPERTS, HIDDEN, 2 * HIDDEN), "bf16"],
        w2: Tensor[(EXPERTS, HIDDEN, HIDDEN), "bf16"],
    ) -> Tensor[(TOKENS, HIDDEN), "bf16"]:
        with Mesh(("cta",), layout=(8,), names=("expert_tile",)) as cta:
            weights, experts = tf.route_topk(x, experts=EXPERTS, k=TOPK)
            routed = tf.expert_gather(x, experts, owner=cta, storage="smem")
            gate_up = tf.grouped_mma(routed, w1, accumulator="rmem")
            hidden = tf.silu_and_mul(gate_up, storage="rmem")
            down = tf.grouped_mma(hidden, w2, accumulator="rmem")
            out = tf.weighted_scatter_reduce(down, weights, experts, storage="smem")
            return tf.reshard(out, (TOKENS, HIDDEN), "gmem")

    @func
    def fused_thread(
        x: Tensor[(TOKENS, HIDDEN), "bf16"],
        w1: Tensor[(EXPERTS, HIDDEN, 2 * HIDDEN), "bf16"],
        w2: Tensor[(EXPERTS, HIDDEN, HIDDEN), "bf16"],
    ) -> Tensor[(TOKENS, HIDDEN), "bf16"]:
        with Mesh(("thread",), layout=(4, 32), names=("warp", "lane")) as thread:
            return tf.fused_moe_fragment(x, w1, w2, owner=thread, storage="rmem")

    @func
    def fused_gpu(
        x: Tensor[(TOKENS, HIDDEN), "bf16"],
        w1: Tensor[(EXPERTS, HIDDEN, 2 * HIDDEN), "bf16"],
        w2: Tensor[(EXPERTS, HIDDEN, HIDDEN), "bf16"],
    ) -> Tensor[(TOKENS, HIDDEN), "bf16"]:
        with Mesh(("gpu",), layout=(8,), names=("rank",)) as gpu:
            routed = tf.all_to_all_expert_dispatch(x, owner=gpu, storage="gmem")
            local = tf.fused_moe(routed, w1, w2, owner=gpu)
            return tf.all_to_all_expert_combine(local, owner=gpu)


# noqa
@module(entry="fused_cta", target=TARGET, topologies=CUDA_TOPOS)
class K01FusedDecode:
    @func
    def unfused(x, conv_weight, state, norm_weight, gate):
        conv = tf.depthwise_conv4(x, conv_weight)
        activated = tf.silu(conv)
        next_state, read = tf.kda_update(state, activated)
        return tf.rms_norm(read, norm_weight) * tf.silu(gate), next_state

    @func
    def fused_cta(x, conv_weight, state, norm_weight, gate):
        with Mesh(("cta",), layout=(8,), names=("head",)) as cta:
            x_local = tf.place(x, owner=cta, storage="smem")
            conv = tf.depthwise_conv4(x_local, conv_weight, storage="smem")
            activated = tf.silu(conv)
            next_state, read = tf.kda_update(state, activated, owner=cta, storage="smem")
            out = tf.rms_norm(read, norm_weight) * tf.silu(gate)
            return tf.reshard(out, tf.logical_layout(out), "gmem"), next_state

    @func
    def fused_thread(x, conv_weight, state, norm_weight, gate):
        with Mesh(("thread",), layout=(128,), names=("channel",)) as thread:
            return tf.fused_kda_decode(
                x, conv_weight, state, norm_weight, gate, owner=thread, storage="rmem"
            )


# noqa
@module(entry="fused_cta", target=TARGET, topologies=CUDA_TOPOS)
class S01SSDCombined:
    @func
    def unfused(x, a, b, c, d, z):
        cumulative = tf.chunk_cumsum(x, a)
        states = tf.ssd_state_passing(cumulative, b)
        scanned = tf.ssd_scan(states, c)
        return tf.silu(z) * (scanned + d * x)

    @func
    def fused_cta(x, a, b, c, d, z):
        with Mesh(("cta",), layout=(8,), names=("chunk",)) as cta:
            cumulative = tf.chunk_cumsum(x, a, owner=cta, storage="smem")
            states = tf.ssd_state_passing(cumulative, b, storage="smem")
            scanned = tf.ssd_scan(states, c, storage="rmem")
            out = tf.silu(z) * (scanned + d * x)
            return tf.reshard(out, tf.logical_layout(out), "gmem")

    @func
    def fused_thread(x, a, b, c, d, z):
        with Mesh(("thread",), layout=(128,), names=("channel",)) as thread:
            return tf.ssd_combined(x, a, b, c, d, z, owner=thread, storage="rmem")


# noqa
@module(entry="fused_cta", target=TARGET, topologies=CUDA_TOPOS)
class H02MHCWithPrenorm:
    @func
    def unfused(x, h, gamma, beta):
        normalized = tf.layer_norm(x, gamma, beta, axis=-1, eps=1e-5)
        transform = tf.sinkhorn_transform(h)
        return tf.residual_mix(normalized, transform)

    @func
    def fused_cta(x, h, gamma, beta):
        with Mesh(("cta",), layout=(8,), names=("row",)) as cta:
            local = tf.place(x, owner=cta, storage="smem")
            normalized = tf.layer_norm(local, gamma, beta, axis=-1, eps=1e-5)
            transform = tf.sinkhorn_transform(h, owner=cta, storage="smem")
            mixed = tf.residual_mix(normalized, transform, storage="smem")
            return tf.reshard(mixed, tf.logical_layout(mixed), "gmem")

    @func
    def fused_thread(x, h, gamma, beta):
        with Mesh(("thread",), layout=(128,), names=("element",)) as thread:
            return tf.mhc_pre_with_prenorm(x, h, gamma, beta, owner=thread, storage="rmem")


# noqa
# noqa
# noqa
