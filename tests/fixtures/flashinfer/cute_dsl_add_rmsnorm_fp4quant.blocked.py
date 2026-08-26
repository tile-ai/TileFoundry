"""Placed CuTe Add-RMSNorm followed by packed NVFP4 quantization.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/cute_dsl/add_rmsnorm_fp4quant.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: DType: unknown value 'nvfp4'
ledger: REG-05, OP-01

The return preserves residual writeback, packed payload, and E4M3 scales.
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
@module(entry="add_rmsnorm_fp4quant", target=TARGET, topologies=TOPOLOGIES)
class AddRMSNormFP4QuantKernel:
    @func
    def add_rmsnorm_fp4quant(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        residual: Tensor[(ROWS, HIDDEN), "bf16"],
        weight: ConstTensor[(HIDDEN,), "bf16"],
        global_scale: ConstTensor[(1,), "f32"],
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
            residual_reg = tf.reshard(
                residual,
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
            residual_smem = tf.reshard(
                residual_reg,
                (
                    ROWS @ (mesh.block, mesh.row),
                    HIDDEN @ (mesh.warp, mesh.lane),
                ),
                "smem",
            )
            weight_smem = tf.reshard(weight, (HIDDEN,), "smem")
            scale_smem = tf.reshard(global_scale, (1,), "smem")
            summed = x_smem + residual_smem
            normalized = tf.rms_norm(summed, weight_smem, eps=1e-6)
            scaled = tf.cast(normalized, "f32") / scale_smem
            packed, block_scales = tf.quant(
                scaled,
                group=16,
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
                        (HIDDEN // 16) @ (mesh.warp, mesh.lane),
                    ),
                    "gmem",
                ),
                tf.reshard(
                    summed,
                    (
                        ROWS @ (mesh.block, mesh.row),
                        HIDDEN @ (mesh.warp, mesh.lane),
                    ),
                    "gmem",
                ),
            )


__all__ = ["AddRMSNormFP4QuantKernel"]
