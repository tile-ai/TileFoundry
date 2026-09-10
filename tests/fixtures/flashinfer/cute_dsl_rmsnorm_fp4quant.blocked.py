"""Placed CuTe RMSNorm followed by packed MXFP4 quantization.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/cute_dsl/rmsnorm_fp4quant.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: DType: unknown value 'nvfp4'
ledger: REG-05, OP-01

The program preserves 32-element MXFP4 blocks and their UE8M0 scale grid.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 128, 4096
CTA_COUNT, ROWS_PER_CTA, WARPS_PER_ROW = 64, 2, 2
TARGET = CudaTarget("nvidia.b200_sxm")
TOPOLOGIES = (
    Topology("cta", CTA_COUNT),
    Topology("thread", ROWS_PER_CTA * WARPS_PER_ROW * 32),
)


# noqa
@module(entry="rmsnorm_fp4quant", target=TARGET, topologies=TOPOLOGIES)
class RMSNormFP4QuantKernel:
    @func
    def rmsnorm_fp4quant(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        weight: ConstTensor[(HIDDEN,), "bf16"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(CTA_COUNT, ROWS_PER_CTA, WARPS_PER_ROW, 32),
            names=("block", "row", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (
                    ROWS @ (mesh.block, mesh.row),
                    HIDDEN @ (mesh.warp, mesh.lane),
                ),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (
                    ROWS @ (mesh.block, mesh.row),
                    HIDDEN @ (mesh.warp, mesh.lane),
                ),
                "smem",
            )
            weight_smem = tf.reshard(weight, (HIDDEN,), "smem")
            normalized = tf.rms_norm(x_smem, weight_smem, eps=1e-6)
            packed, block_scales = tf.quant(
                tf.cast(normalized, "f32"),
                group=32,
                target_dtype="nvfp4",
            )
            return (
                tf.reshard(
                    packed,
                    (
                        ROWS @ (mesh.block, mesh.row),
                        HIDDEN @ (mesh.warp, mesh.lane),
                    ),
                    "gmem",
                ),
                tf.reshard(
                    block_scales,
                    (
                        ROWS @ (mesh.block, mesh.row),
                        (HIDDEN // 32) @ (mesh.warp, mesh.lane),
                    ),
                    "gmem",
                ),
            )


__all__ = ["RMSNormFP4QuantKernel"]
