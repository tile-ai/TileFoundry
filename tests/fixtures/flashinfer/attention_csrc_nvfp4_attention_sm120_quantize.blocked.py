"""Block-scaled NVFP4 quantization for SM120 attention operands.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
csrc/nvfp4_attention_sm120/nvfp4_attention_sm120_quantize.cu
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: DType: unknown value 'nvfp4'
ledger: REG-05, OP-01

The specialization keeps 16-token blocks, packed half-width payloads, and one
E4M3 scale per 16 input elements for normal and transposed layouts.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, TOKENS, HEADS, HEAD_DIM = 2, 128, 32, 128
BLOCK, CTA_COUNT, THREAD_COUNT = 16, BATCH * HEADS * (TOKENS // 16), 128
TARGET = CudaTarget("nvidia.b200_sxm")
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="quantize", target=TARGET, topologies=TOPOLOGIES)
class ScaledFP4QuantKernel:
    """FlashInfer scaled_fp4_quant_kernel entry."""

    @func
    def quantize(x: Tensor[(BATCH, TOKENS, HEADS, HEAD_DIM), "bf16"]):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, HEADS, TOKENS // BLOCK, 8, 16),
            names=("batch", "head", "token_block", "token", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (
                    BATCH @ mesh.batch,
                    TOKENS @ (mesh.token_block, mesh.token),
                    HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (
                    BATCH @ mesh.batch,
                    TOKENS @ (mesh.token_block, mesh.token),
                    HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            packed, scales = tf.quant(tf.cast(x_smem, "f32"), group=16, target_dtype="nvfp4")
            return packed, scales


@module(entry="quantize", target=TARGET, topologies=TOPOLOGIES)
class ScaledFP4QuantTransKernel:
    """FlashInfer scaled_fp4_quant_trans_kernel entry."""

    kernel = ScaledFP4QuantKernel.renamed("kernel")

    @func
    def quantize(x: Tensor[(BATCH, TOKENS, HEADS, HEAD_DIM), "bf16"]):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, HEADS, TOKENS // BLOCK, 8, 16),
            names=("batch", "head", "token_block", "token", "lane"),
        ) as mesh:
            staged = tf.reshard(
                x,
                (
                    BATCH @ mesh.batch,
                    TOKENS @ (mesh.token_block, mesh.token),
                    HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "gmem",
            )
            packed, scales = kernel(staged)
            return (
                tf.transpose(packed, perm=(0, 2, 1, 3)),
                tf.transpose(scales, perm=(0, 2, 1, 3)),
            )


__all__ = ["ScaledFP4QuantKernel", "ScaledFP4QuantTransKernel"]
