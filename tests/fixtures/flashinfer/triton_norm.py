"""Placed public Triton RMSNorm wrappers.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/triton/norm.py
license: Apache-2.0 (no upstream source is vendored)

The two independent modules preserve the plain output API and the residual
wrapper's paired output and in-place residual effects.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 128, 2048
TARGET = CudaTarget("nvidia.h200_sxm")
PLAIN_TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 8 * 32))
RESIDUAL_TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 32 * 32))


@module(entry="rms_norm", target=TARGET, topologies=PLAIN_TOPOLOGIES)
class TritonRMSNorm:
    """FlashInfer public Triton RMSNorm wrapper kernel.

    predicted-ns: 372  waves: 1
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def rms_norm(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        weight: ConstTensor[(HIDDEN,), "bf16"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 8, 32),
            names=("row", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x, (ROWS @ mesh.row, 8 @ mesh.warp, 256 @ mesh.lane), "rmem"
            )
            x_smem = tf.reshard(
                x_reg, (ROWS @ mesh.row, 8 @ mesh.warp, 256 @ mesh.lane), "smem"
            )
            weight_smem = tf.reshard(weight, (HIDDEN,), "smem")
            normalized = tf.rms_norm(x_smem, weight_smem, eps=1e-6)
            return tf.reshard(
                normalized,
                (ROWS @ mesh.row, 8 @ mesh.warp, 256 @ mesh.lane),
                "gmem",
            )


@module(entry="rms_norm_add_residual", target=TARGET, topologies=RESIDUAL_TOPOLOGIES)
class TritonRMSNormAddResidual:
    """FlashInfer public Triton residual-add RMSNorm wrapper kernel.

    predicted-ns: 599  waves: 1
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def rms_norm_add_residual(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        residual: Tensor[(ROWS, HIDDEN), "bf16"],
        weight: ConstTensor[(HIDDEN,), "bf16"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 32, 32),
            names=("row", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x, (ROWS @ mesh.row, 64 @ mesh.warp, 32 @ mesh.lane), "rmem"
            )
            residual_reg = tf.reshard(
                residual,
                (ROWS @ mesh.row, 64 @ mesh.warp, 32 @ mesh.lane),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg, (ROWS @ mesh.row, 64 @ mesh.warp, 32 @ mesh.lane), "smem"
            )
            residual_smem = tf.reshard(
                residual_reg,
                (ROWS @ mesh.row, 64 @ mesh.warp, 32 @ mesh.lane),
                "smem",
            )
            weight_smem = tf.reshard(weight, (HIDDEN,), "smem")
            summed = x_smem + residual_smem
            normalized = tf.rms_norm(summed, weight_smem, eps=1e-6)
            return (
                tf.reshard(
                    normalized,
                    (ROWS @ mesh.row, 64 @ mesh.warp, 32 @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(
                    summed,
                    (ROWS @ mesh.row, 64 @ mesh.warp, 32 @ mesh.lane),
                    "gmem",
                ),
            )


__all__ = ["TritonRMSNorm", "TritonRMSNormAddResidual"]
