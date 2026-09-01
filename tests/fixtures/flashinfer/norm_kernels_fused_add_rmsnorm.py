"""Placed CuTe fused residual-add and RMSNorm kernels.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/norm/kernels/fused_add_rmsnorm.py
license: Apache-2.0 (no upstream source is vendored)

Each 128-thread CTA owns four rows, one row per warp, matching the CuTe
``rows_per_block`` dispatch for hidden size 1024. Functional tuple returns
preserve the upstream in-place writes.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 256, 1024
CTA_COUNT, ROWS_PER_CTA = 64, 4
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", ROWS_PER_CTA * 32))


@module(entry="fused_add_rmsnorm_cute", target=TARGET, topologies=TOPOLOGIES)
class FusedAddRMSNormKernel:
    """FlashInfer CuTe fused residual-add and RMSNorm kernel.

    predicted-ns: 1027  waves: 1
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def fused_add_rmsnorm_cute(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        residual: Tensor[(ROWS, HIDDEN), "bf16"],
        weight: ConstTensor[(HIDDEN,), "bf16"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(CTA_COUNT, ROWS_PER_CTA, 32),
            names=("block", "row", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "rmem",
            )
            residual_reg = tf.reshard(
                residual,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "smem",
            )
            residual_smem = tf.reshard(
                residual_reg,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "smem",
            )
            weight_smem = tf.reshard(weight, (HIDDEN,), "smem")
            summed = x_smem + residual_smem
            normalized = tf.rms_norm(summed, weight_smem, eps=1e-6)
            return (
                tf.reshard(
                    normalized,
                    (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(
                    summed,
                    (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                    "gmem",
                ),
            )


@module(entry="fused_add_rmsnorm_quant_cute", target=TARGET, topologies=TOPOLOGIES)
class FusedAddRMSNormQuantKernel:
    """FlashInfer CuTe fused residual-add, RMSNorm, and FP8 kernel.

    predicted-ns: 934  waves: 1
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def fused_add_rmsnorm_quant_cute(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        residual: Tensor[(ROWS, HIDDEN), "bf16"],
        weight: ConstTensor[(HIDDEN,), "bf16"],
        scale: ConstTensor[(1,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(CTA_COUNT, ROWS_PER_CTA, 32),
            names=("block", "row", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "rmem",
            )
            residual_reg = tf.reshard(
                residual,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "smem",
            )
            residual_smem = tf.reshard(
                residual_reg,
                (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                "smem",
            )
            weight_smem = tf.reshard(weight, (HIDDEN,), "smem")
            scale_smem = tf.reshard(scale, (1,), "smem")
            summed = x_smem + residual_smem
            normalized = tf.rms_norm(summed, weight_smem, eps=1e-6)
            quantized = tf.cast(tf.cast(normalized, "f32") / scale_smem, "fp8e4m3")
            return (
                tf.reshard(
                    quantized,
                    (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(
                    summed,
                    (ROWS @ (mesh.block, mesh.row), HIDDEN @ mesh.lane),
                    "gmem",
                ),
            )


__all__ = ["FusedAddRMSNormKernel", "FusedAddRMSNormQuantKernel"]
