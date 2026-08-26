"""Placed CUDA RMSNorm followed by fixed-scale FP8 quantization.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
csrc/norm.cu:rmsnorm_quant
license: Apache-2.0 (no upstream source is vendored)

Each CTA owns one token row. Four warps and 32 lanes vectorize the hidden
dimension, matching the row-wise CUDA normalization kernel.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 256, 1536
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 4 * 32))


@module(entry="norm_quant", target=TARGET, topologies=TOPOLOGIES)
class RMSNormQuant:
    """FlashInfer CUDA fixed-scale FP8 RMSNorm kernel.

    predicted-ns: 496  waves: 2
    measured-ns: 11916 (flashinfer 0.6.18, NVIDIA H200, 2026-08-25)
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def norm_quant(
        a: Tensor[(ROWS, HIDDEN), "bf16"],
        weight: ConstTensor[(HIDDEN,), "bf16"],
        scale: ConstTensor[(1,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 4, 32),
            names=("row", "warp", "lane"),
        ) as mesh:
            a_reg = tf.reshard(
                a,
                (ROWS @ mesh.row, 12 @ mesh.warp, 128 @ mesh.lane),
                "rmem",
            )
            weight_reg = tf.reshard(weight, (HIDDEN,), "rmem")
            a_smem = tf.reshard(
                a_reg,
                (ROWS @ mesh.row, 12 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            weight_smem = tf.reshard(weight_reg, (HIDDEN,), "smem")
            scale_smem = tf.reshard(scale, (1,), "smem")
            normalized = tf.rms_norm(a_smem, weight_smem, eps=1e-6)
            quantized = tf.cast(tf.cast(normalized, "f32") / scale_smem, "fp8e4m3")
            return tf.reshard(
                quantized,
                (ROWS @ mesh.row, 12 @ mesh.warp, 128 @ mesh.lane),
                "gmem",
            )


__all__ = ["RMSNormQuant"]
