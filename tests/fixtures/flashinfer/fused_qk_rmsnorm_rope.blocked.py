"""Placed fused QK RMSNorm and three-dimensional RoPE kernel.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 \
include/flashinfer/norm/fused_qk_rmsnorm_rope.cuh
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: Split: Split: axis 1 is divided into 3 parts and is already Split across participants
ledger: REG-08

The program keeps the packed QKV input, across-head RMSNorm, derived 3D
positions, and the upstream three-output contract.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

TOKENS, HEADS, HEAD_DIM = 120, 24, 128
PPF, PPH, PPW = 1, 5, 24
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", TOKENS * 3), Topology("thread", HEADS * 32))


@module(entry="fused_qk_norm_rope", target=TARGET, topologies=TOPOLOGIES)
class FusedQKNormRopeKernel:
    @func
    def fused_qk_norm_rope(
        qkv: Tensor[(TOKENS, 3 * HEADS * HEAD_DIM), "bf16"],
        q_weight: ConstTensor[(HEADS * HEAD_DIM,), "bf16"],
        k_weight: ConstTensor[(HEADS * HEAD_DIM,), "bf16"],
        freq: ConstTensor[(HEAD_DIM // 2,), "f32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(TOKENS, 3, HEADS, 32),
            names=("token", "kind", "head", "lane"),
        ) as mesh:
            qkv_reg = tf.reshard(
                qkv,
                (
                    TOKENS @ mesh.token,
                    3,
                    HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            qkv_smem = tf.reshard(
                qkv_reg,
                (
                    TOKENS @ mesh.token,
                    3,
                    HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            q_smem, k_smem, v_smem = tf.split(qkv_smem, axis=1, num_splits=3)
            q_weight_smem = tf.reshard(q_weight, (HEADS * HEAD_DIM,), "smem")
            k_weight_smem = tf.reshard(k_weight, (HEADS * HEAD_DIM,), "smem")
            freq_smem = tf.reshard(freq, (HEAD_DIM // 2,), "smem")
            positions = tf.arange(Tensor[(TOKENS,), "i64"])
            frame_reg = tf.reshard(
                positions // (PPH * PPW), (TOKENS @ mesh.token,), "rmem"
            )
            height_reg = tf.reshard(
                (positions % (PPH * PPW)) // PPW,
                (TOKENS @ mesh.token,),
                "rmem",
            )
            width_reg = tf.reshard(
                positions % PPW, (TOKENS @ mesh.token,), "rmem"
            )
            q_normalized_flat = tf.rms_norm(q_smem, q_weight_smem, eps=1e-6)
            k_normalized_flat = tf.rms_norm(k_smem, k_weight_smem, eps=1e-6)
            q_normalized = tf.reshape(
                q_normalized_flat,
                (TOKENS, HEADS, HEAD_DIM),
            )
            k_normalized = tf.reshape(
                k_normalized_flat,
                (TOKENS, HEADS, HEAD_DIM),
            )
            v = tf.reshape(v_smem, (TOKENS, HEADS, HEAD_DIM))
            q_rope, k_rope = tf.rope_3d(
                q_normalized,
                k_normalized,
                freq_smem,
                frame_reg,
                height_reg,
                width_reg,
            )
            return (
                tf.reshard(
                    q_rope,
                    (TOKENS @ mesh.token, HEADS @ mesh.head, HEAD_DIM @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(
                    k_rope,
                    (TOKENS @ mesh.token, HEADS @ mesh.head, HEAD_DIM @ mesh.lane),
                    "gmem",
                ),
                tf.reshard(
                    v,
                    (TOKENS @ mesh.token, HEADS @ mesh.head, HEAD_DIM @ mesh.lane),
                    "gmem",
                ),
            )


__all__ = ["FusedQKNormRopeKernel"]
