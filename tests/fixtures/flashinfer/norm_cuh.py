"""Placed CUDA general LayerNorm kernel with a split reduction axis.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
include/flashinfer/norm.cuh
license: Apache-2.0 (no upstream source is vendored)
The specialization keeps gamma, beta, FP8 output, and hidden-axis vectorization
while spelling out sum and sumsq reduction state.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 256, 1024
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 32 * 32))


@module(entry="general_layer_norm", target=TARGET, topologies=TOPOLOGIES)
class GeneralLayerNormKernel:
    """FlashInfer GeneralLayerNorm kernel.

    predicted-ns: 700
    waves: 2
    measured-ns: not taken
    note: recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def general_layer_norm(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        gamma: ConstTensor[(HIDDEN,), "f32"],
        beta: ConstTensor[(HIDDEN,), "f32"],
        scale: ConstTensor[(1,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 32, 32),
            names=("row", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x, (ROWS @ mesh.row, 32 @ mesh.warp, 32 @ mesh.lane), "rmem"
            )
            x_smem = tf.reshard(
                x_reg, (ROWS @ mesh.row, 32 @ mesh.warp, 32 @ mesh.lane), "smem"
            )
            gamma_smem = tf.reshard(gamma, (HIDDEN,), "smem")
            beta_smem = tf.reshard(beta, (HIDDEN,), "smem")
            scale_smem = tf.reshard(scale, (1,), "smem")
            x32 = tf.cast(x_smem, "f32")
            count = tf.reduce(
                tf.full_like(x32, value=1.0), axes=(-1,), keepdim=True, kind="sum"
            )
            summed = tf.reduce(x32, axes=(-1,), keepdim=True, kind="sum")
            sumsq = tf.reduce(x32 * x32, axes=(-1,), keepdim=True, kind="sum")
            mean = summed / count
            variance = sumsq / count - mean * mean
            normalized = (x32 - mean) * tf.rsqrt(variance + 1e-6)
            affine = normalized * gamma_smem + beta_smem
            quantized = tf.cast(affine / scale_smem, "fp8e4m3")
            return tf.reshard(
                quantized,
                (ROWS @ mesh.row, 32 @ mesh.warp, 32 @ mesh.lane),
                "gmem",
            )


__all__ = ["GeneralLayerNormKernel"]
