"""Placed Q01 SwiGLU followed by packed NVFP4 block quantization.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/quantization/fp4_quantization.py:silu_and_mul_nvfp4_quantize
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: DType: unknown value 'nvfp4'
ledger: REG-05, OP-01

Token CTAs and vectorized threads preserve the packed NVFP4 payload and \
per-16-element scales; substituting FP8 Quant would change the kernel.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN = 256, 1024
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", ROWS), Topology("thread", 4 * 32))


# noqa
@module(entry="silu_and_mul_nvfp4_quantize", target=TARGET, topologies=TOPOLOGIES)
class SwiGLUNVFP4Quantize:
    @func
    def silu_and_mul_nvfp4_quantize(
        gate: Tensor[(ROWS, HIDDEN), "bf16"],
        up: Tensor[(ROWS, HIDDEN), "bf16"],
        global_scale: ConstTensor[(1,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(ROWS, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            gate_reg = tf.reshard(
                gate,
                (ROWS @ mesh.token, 8 @ mesh.warp, 128 @ mesh.lane),
                "rmem",
            )
            up_reg = tf.reshard(
                up,
                (ROWS @ mesh.token, 8 @ mesh.warp, 128 @ mesh.lane),
                "rmem",
            )
            gate_smem = tf.reshard(
                gate_reg,
                (ROWS @ mesh.token, 8 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            up_smem = tf.reshard(
                up_reg,
                (ROWS @ mesh.token, 8 @ mesh.warp, 128 @ mesh.lane),
                "smem",
            )
            scale_smem = tf.reshard(global_scale, (1,), "smem")
            activated = tf.silu(gate_smem) * up_smem
            scaled = tf.cast(activated, "f32") / scale_smem
            quantized, scales = tf.quant(
                scaled,
                group=16,
                target_dtype="nvfp4",
            )
            return (
                tf.reshard(
                    quantized,
                    (ROWS @ mesh.token, 8 @ mesh.warp, 128 @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(
                    scales,
                    (ROWS @ mesh.token, 8 @ mesh.warp),
                    "gmem",
                ),
            )


__all__ = ["SwiGLUNVFP4Quantize"]
