"""Placed gated DiT residual LayerNorm scale-shift kernel.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
include/flashinfer/norm/fused_dit_layernorm.cuh
license: Apache-2.0 (no upstream source is vendored)
The specialization keeps WAN's gate, scale, shift, and residual writeback while
spelling out sum and sumsq reduction state.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 256, 3072
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 12 * 32))


@module(entry="meta_fused_layernorm", target=TARGET, topologies=TOPOLOGIES)
class MetaFusedLayerNormKernel:
    """FlashInfer fused DiT LayerNorm scale-shift kernel.

    predicted-ns: 4676
    waves: 2
    measured-ns: not taken
    note: recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def meta_fused_layernorm(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        residual: Tensor[(ROWS, HIDDEN), "bf16"],
        gate: Tensor[(ROWS, HIDDEN), "bf16"],
        gate_bias: ConstTensor[(HIDDEN,), "f32"],
        scale: Tensor[(ROWS, HIDDEN), "bf16"],
        scale_bias: ConstTensor[(HIDDEN,), "f32"],
        shift: Tensor[(ROWS, HIDDEN), "bf16"],
        shift_bias: ConstTensor[(HIDDEN,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 12, 32),
            names=("row", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x, (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane), "rmem"
            )
            residual_reg = tf.reshard(
                residual,
                (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane),
                "rmem",
            )
            gate_reg = tf.reshard(
                gate, (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane), "rmem"
            )
            scale_reg = tf.reshard(
                scale, (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane), "rmem"
            )
            shift_reg = tf.reshard(
                shift, (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane), "rmem"
            )
            x_smem = tf.reshard(
                x_reg, (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane), "smem"
            )
            residual_smem = tf.reshard(
                residual_reg,
                (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            gate_smem = tf.reshard(
                gate_reg,
                (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            scale_smem = tf.reshard(
                scale_reg,
                (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            shift_smem = tf.reshard(
                shift_reg,
                (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            gate_bias_smem = tf.reshard(gate_bias, (HIDDEN,), "smem")
            scale_bias_smem = tf.reshard(scale_bias, (HIDDEN,), "smem")
            shift_bias_smem = tf.reshard(shift_bias, (HIDDEN,), "smem")
            gated = tf.cast(x_smem, "f32") * (
                tf.cast(gate_smem, "f32") + gate_bias_smem
            )
            summed = gated + tf.cast(residual_smem, "f32")
            count = tf.reduce(
                tf.full_like(summed, value=1.0), axes=(-1,), keepdim=True, kind="sum"
            )
            reduced = tf.reduce(summed, axes=(-1,), keepdim=True, kind="sum")
            sumsq = tf.reduce(summed * summed, axes=(-1,), keepdim=True, kind="sum")
            mean = reduced / count
            variance = sumsq / count - mean * mean
            normalized = (summed - mean) * tf.rsqrt(variance + 1e-6)
            shifted = normalized * (
                tf.cast(scale_smem, "f32") + scale_bias_smem + 1.0
            ) + (tf.cast(shift_smem, "f32") + shift_bias_smem)
            return (
                tf.reshard(
                    tf.cast(shifted, "bf16"),
                    (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(
                    tf.cast(summed, "bf16"),
                    (ROWS @ mesh.row, 24 @ mesh.warp, 128 @ mesh.lane),
                    "gmem",
                ),
            )


__all__ = ["MetaFusedLayerNormKernel"]
