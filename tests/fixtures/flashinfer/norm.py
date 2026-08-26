"""Placed FlashInfer LayerNorm fusion boundaries blocked by reduction sharding.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/norm/__init__.py
license: Apache-2.0 (no upstream source is vendored)
The explicit sum and sumsq reductions preserve token-CTA and hidden-thread
placement without hiding cross-participant combination inside LayerNorm.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN, DIT_HIDDEN = 256, 1024, 3072
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 4 * 32))


@module(entry="layernorm_quant", target=TARGET, topologies=TOPOLOGIES)
class LayerNormFixedScaleFP8:
    """FlashInfer rmsnorm_fp8_quant kernel.

    predicted-ns: 700
    waves: 2
    measured-ns: not taken
    note: recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def layernorm_quant(
        x: Tensor[(ROWS, HIDDEN), "bf16"],
        gamma: ConstTensor[(HIDDEN,), "f32"],
        beta: ConstTensor[(HIDDEN,), "f32"],
        scale: ConstTensor[(1,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (ROWS @ mesh.token, 8 @ mesh.warp, 128 @ mesh.lane),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (ROWS @ mesh.token, 8 @ mesh.warp, 128 @ mesh.lane),
                "smem",
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
                (ROWS @ mesh.token, 8 @ mesh.warp, 128 @ mesh.lane),
                "gmem",
            )


@module(entry="fused_dit_gate_layernorm", target=TARGET, topologies=TOPOLOGIES)
class DiTGateLayerNorm:
    """FlashInfer fused DiT gate and LayerNorm kernel.

    predicted-ns: 4266
    waves: 2
    measured-ns: not taken
    note: recorded evidence, not an assertion; open a ledger row if predicted exceeds measured.
    """

    @func
    def fused_dit_gate_layernorm(
        x: Tensor[(ROWS, DIT_HIDDEN), "bf16"],
        residual: Tensor[(ROWS, DIT_HIDDEN), "bf16"],
        gate: Tensor[(ROWS, DIT_HIDDEN), "bf16"],
        gate_bias: ConstTensor[(DIT_HIDDEN,), "f32"],
        gamma: ConstTensor[(DIT_HIDDEN,), "f32"],
        beta: ConstTensor[(DIT_HIDDEN,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (ROWS @ mesh.token, 24 @ mesh.warp, 128 @ mesh.lane),
                "rmem",
            )
            residual_reg = tf.reshard(
                residual,
                (ROWS @ mesh.token, 24 @ mesh.warp, 128 @ mesh.lane),
                "rmem",
            )
            gate_reg = tf.reshard(
                gate,
                (ROWS @ mesh.token, 24 @ mesh.warp, 128 @ mesh.lane),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (ROWS @ mesh.token, 24 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            residual_smem = tf.reshard(
                residual_reg,
                (ROWS @ mesh.token, 24 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            gate_smem = tf.reshard(
                gate_reg,
                (ROWS @ mesh.token, 24 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            gate_bias_smem = tf.reshard(gate_bias, (DIT_HIDDEN,), "smem")
            gamma_smem = tf.reshard(gamma, (DIT_HIDDEN,), "smem")
            beta_smem = tf.reshard(beta, (DIT_HIDDEN,), "smem")
            gate_f32 = tf.cast(gate_smem, "f32") + gate_bias_smem
            summed = tf.cast(residual_smem, "f32") + tf.cast(x_smem, "f32") * gate_f32
            count = tf.reduce(
                tf.full_like(summed, value=1.0), axes=(-1,), keepdim=True, kind="sum"
            )
            reduced = tf.reduce(summed, axes=(-1,), keepdim=True, kind="sum")
            sumsq = tf.reduce(summed * summed, axes=(-1,), keepdim=True, kind="sum")
            mean = reduced / count
            variance = sumsq / count - mean * mean
            normalized = (summed - mean) * tf.rsqrt(variance + 1e-6)
            normalized = normalized * gamma_smem + beta_smem
            return (
                tf.reshard(
                    tf.cast(summed, "bf16"),
                    (ROWS @ mesh.token, 24 @ mesh.warp, 128 @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(
                    normalized,
                    (ROWS @ mesh.token, 24 @ mesh.warp, 128 @ mesh.lane),
                    "gmem",
                ),
            )


__all__ = ["LayerNormFixedScaleFP8", "DiTGateLayerNorm"]
