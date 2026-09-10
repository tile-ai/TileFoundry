"""Placed G01 NVFP4 residual GEMM with LoRA-up and bias epilogue.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
flashinfer/gemm/gemm_svdquant.py:mm_nvfp4_svdquant
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: DType: unknown value 'nvfp4'
ledger: EXT-04, OP-01, OP-02

Row CTAs and vectorized threads preserve packed operands, block scales, \
BF16 LoRA correction, alpha, and bias instead of substituting a BF16 twin.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

M, K, N, R = 256, 128, 128, 32
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", M), Topology("thread", 4 * 32))


# noqa
@module(entry="mm_nvfp4_svdquant", target=TARGET, topologies=TOPOLOGIES)
class NVFP4SVDQuantGemm:
    @func
    def mm_nvfp4_svdquant(
        a: Tensor[(M, K), "nvfp4"],
        b: ConstTensor[(N, K), "nvfp4"],
        a_scale: Tensor[(M, K // 16), "fp8e4m3"],
        b_scale: ConstTensor[(N, K // 16), "fp8e4m3"],
        alpha: ConstTensor[(1,), "f32"],
        down: Tensor[(M, R), "bf16"],
        lora_up: ConstTensor[(N, R), "bf16"],
        bias: ConstTensor[(N,), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:
        with Mesh(
            ("cta", "thread"),
            layout=(M, 4, 32),
            names=("row", "warp", "lane"),
        ) as mesh:
            a_reg = tf.reshard(
                a,
                (M @ mesh.row, 1 @ mesh.warp, K @ mesh.lane),
                "rmem",
            )
            b_reg = tf.reshard(
                b,
                (N, 1 @ mesh.warp, K @ mesh.lane),
                "rmem",
            )
            a_smem = tf.reshard(
                a_reg,
                (M @ mesh.row, 1 @ mesh.warp, K @ mesh.lane),
                "smem",
            )
            b_smem = tf.reshard(
                b_reg,
                (N, 1 @ mesh.warp, K @ mesh.lane),
                "smem",
            )
            a_scale_smem = tf.reshard(a_scale, (M @ mesh.row, K // 16), "smem")
            b_scale_smem = tf.reshard(b_scale, (N, K // 16), "smem")
            down_smem = tf.reshard(down, (M @ mesh.row, R), "smem")
            lora_up_smem = tf.reshard(lora_up, (N, R), "smem")
            bias_smem = tf.reshard(bias, (N,), "smem")
            residual = tf.block_scaled_matmul(
                a_smem,
                tf.transpose(b_smem, perm=(1, 0)),
                a_scale_smem,
                b_scale_smem,
            )
            correction = down_smem @ tf.transpose(lora_up_smem, perm=(1, 0))
            output = alpha * residual + correction + bias_smem
            return tf.reshard(
                output,
                (M @ mesh.row, 4 @ mesh.warp, 32 @ mesh.lane),
                "gmem",
            )


__all__ = ["NVFP4SVDQuantGemm"]
