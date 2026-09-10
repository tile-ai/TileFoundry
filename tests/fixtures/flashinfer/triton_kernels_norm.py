"""Placed Triton RMSNorm kernel with residual and scaling specializations.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/triton/kernels/norm.py
license: Apache-2.0 (no upstream source is vendored)

The concrete specialization keeps input dequantization, residual writeback,
FP8 output scaling, and one row per CTA from the parameterized Triton kernel.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 128, 2048
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 32 * 32))


@module(entry="rms_norm_kernel", target=TARGET, topologies=TOPOLOGIES)
class TritonRMSNormKernel:
    """FlashInfer Triton scaled residual RMSNorm kernel.

    predicted-ns: 521  waves: 1
    measured-ns: not taken
    note: Recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def rms_norm_kernel(
        x: Tensor[(ROWS, HIDDEN), "fp8e4m3"],
        residual: Tensor[(ROWS, HIDDEN), "bf16"],
        weight: ConstTensor[(HIDDEN,), "bf16"],
        input_scale: ConstTensor[(1,), "f32"],
        output_scale: ConstTensor[(1,), "f32"],
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
            input_scale_smem = tf.reshard(input_scale, (1,), "smem")
            output_scale_smem = tf.reshard(output_scale, (1,), "smem")
            dequantized = tf.cast(x_smem, "f32") * input_scale_smem
            summed = dequantized + tf.cast(residual_smem, "f32")
            normalized = tf.rms_norm(
                tf.cast(summed, "bf16"),
                weight_smem,
                eps=1e-6,
            )
            quantized = tf.cast(tf.cast(normalized, "f32") * output_scale_smem, "fp8e4m3")
            return (
                tf.reshard(
                    quantized,
                    (ROWS @ mesh.row, 64 @ mesh.warp, 32 @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(
                    tf.cast(summed, "bf16"),
                    (ROWS @ mesh.row, 64 @ mesh.warp, 32 @ mesh.lane),
                    "gmem",
                ),
            )


__all__ = ["TritonRMSNormKernel"]
