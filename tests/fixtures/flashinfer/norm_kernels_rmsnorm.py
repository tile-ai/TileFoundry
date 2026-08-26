"""Placed CuTe RMSNorm, QK RMSNorm, and fixed-scale FP8 kernels.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/norm/kernels/rmsnorm.py
license: Apache-2.0 (no upstream source is vendored)

The selected specializations preserve the CuTe multi-row dispatch: independent
warps own rows or heads, while lanes own the reduction dimension. The QK
variant retains its three-dimensional batch, head, and head-dimension layout.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

RMS_ROWS, RMS_HIDDEN = 128, 1024
QUANT_ROWS, QUANT_HIDDEN = 64, 4096
BATCH, HEADS, HEAD_DIM = 4, 32, 128
RMS_CTA_COUNT, RMS_ROWS_PER_CTA = 32, 4
QUANT_CTA_COUNT, QUANT_ROWS_PER_CTA = 32, 2
QK_CTA_COUNT, QK_ROWS_PER_CTA = 16, 8
TARGET = CudaTarget("nvidia.h200_sxm")
RMS_TOPOLOGIES = (Topology("cta", RMS_CTA_COUNT), Topology("thread", 4 * 32))
QUANT_TOPOLOGIES = (Topology("cta", QUANT_CTA_COUNT), Topology("thread", 4 * 32))
QK_TOPOLOGIES = (Topology("cta", QK_CTA_COUNT), Topology("thread", 4 * 32))


@module(entry="rmsnorm_cute", target=TARGET, topologies=RMS_TOPOLOGIES)
class RMSNormKernel:
    """FlashInfer CuTe row-wise RMSNorm kernel.

    predicted-ns: 574  waves: 1
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def rmsnorm_cute(
        x: Tensor[(RMS_ROWS, RMS_HIDDEN), "bf16"],
        weight: ConstTensor[(RMS_HIDDEN,), "bf16"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(RMS_CTA_COUNT, RMS_ROWS_PER_CTA, 32),
            names=("block", "row", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (RMS_ROWS @ (mesh.block, mesh.row), RMS_HIDDEN @ mesh.lane),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (RMS_ROWS @ (mesh.block, mesh.row), RMS_HIDDEN @ mesh.lane),
                "smem",
            )
            weight_smem = tf.reshard(weight, (RMS_HIDDEN,), "smem")
            normalized = tf.rms_norm(x_smem, weight_smem, eps=1e-6)
            return tf.reshard(
                normalized,
                (RMS_ROWS @ (mesh.block, mesh.row), RMS_HIDDEN @ mesh.lane),
                "gmem",
            )


@module(entry="qk_rmsnorm_cute", target=TARGET, topologies=QK_TOPOLOGIES)
class QKRMSNormKernel:
    """FlashInfer CuTe three-dimensional QK RMSNorm kernel.

    predicted-ns: 139  waves: 1
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def qk_rmsnorm_cute(
        x: Tensor[(BATCH, HEADS, HEAD_DIM), "bf16"],
        weight: ConstTensor[(HEAD_DIM,), "bf16"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, HEADS // QK_ROWS_PER_CTA, QK_ROWS_PER_CTA, 16),
            names=("batch", "head_group", "row", "lane_in_row"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (
                    BATCH @ mesh.batch,
                    HEADS @ (mesh.head_group, mesh.row),
                    HEAD_DIM @ mesh.lane_in_row,
                ),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (
                    BATCH @ mesh.batch,
                    HEADS @ (mesh.head_group, mesh.row),
                    HEAD_DIM @ mesh.lane_in_row,
                ),
                "smem",
            )
            weight_smem = tf.reshard(weight, (HEAD_DIM,), "smem")
            normalized = tf.rms_norm(x_smem, weight_smem, eps=1e-6)
            return tf.reshard(
                normalized,
                (
                    BATCH @ mesh.batch,
                    HEADS @ (mesh.head_group, mesh.row),
                    HEAD_DIM @ mesh.lane_in_row,
                ),
                "gmem",
            )


@module(entry="rmsnorm_quant_cute", target=TARGET, topologies=QUANT_TOPOLOGIES)
class RMSNormQuantKernel:
    """FlashInfer CuTe RMSNorm and fixed-scale FP8 kernel.

    predicted-ns: 1069  waves: 1
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def rmsnorm_quant_cute(
        x: Tensor[(QUANT_ROWS, QUANT_HIDDEN), "bf16"],
        weight: ConstTensor[(QUANT_HIDDEN,), "bf16"],
        scale: ConstTensor[(1,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(QUANT_CTA_COUNT, QUANT_ROWS_PER_CTA, 2, 32),
            names=("block", "row", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (
                    QUANT_ROWS @ (mesh.block, mesh.row),
                    QUANT_HIDDEN @ (mesh.warp, mesh.lane),
                ),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (
                    QUANT_ROWS @ (mesh.block, mesh.row),
                    QUANT_HIDDEN @ (mesh.warp, mesh.lane),
                ),
                "smem",
            )
            weight_smem = tf.reshard(weight, (QUANT_HIDDEN,), "smem")
            scale_smem = tf.reshard(scale, (1,), "smem")
            normalized = tf.rms_norm(x_smem, weight_smem, eps=1e-6)
            quantized = tf.cast(tf.cast(normalized, "f32") / scale_smem, "fp8e4m3")
            return tf.reshard(
                quantized,
                (
                    QUANT_ROWS @ (mesh.block, mesh.row),
                    QUANT_HIDDEN @ (mesh.warp, mesh.lane),
                ),
                "gmem",
            )


__all__ = ["RMSNormKernel", "QKRMSNormKernel", "RMSNormQuantKernel"]
