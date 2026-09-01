"""Placed rotary-position attention boundary with rank-three inputs.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 flashinfer/rope.py
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: runtime_expression: unsupported call 'tf.rope_with_positions' (5 positional, no keywords)
ledger: EXT-02

Each SURVEY entry remains an independent Module with the source group's
representative supported specialization and upstream mechanism intact.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 4, 128, 2048
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
TARGET = CudaTarget("nvidia.h200_sxm")
CTA_COUNT, THREAD_COUNT = BATCH * QUERY, 4 * 32
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class RopeQuantizeFp8Module1:
    """FlashInfer rope_quantize_fp8 entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class MlaRopeQuantizeFp8Module2:
    """FlashInfer mla_rope_quantize_fp8 entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class RopeQuantizeFp8AppendPagedKvCacheModule3:
    """FlashInfer rope_quantize_fp8_append_paged_kv_cache entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyRopeInplaceModule4:
    """FlashInfer apply_rope_inplace entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyRopePosIdsInplaceModule5:
    """FlashInfer apply_rope_pos_ids_inplace entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyLlama31RopeInplaceModule6:
    """FlashInfer apply_llama31_rope_inplace entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyLlama31RopePosIdsInplaceModule7:
    """FlashInfer apply_llama31_rope_pos_ids_inplace entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyRopeModule8:
    """FlashInfer apply_rope entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyRopePosIdsModule9:
    """FlashInfer apply_rope_pos_ids entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyLlama31RopeModule10:
    """FlashInfer apply_llama31_rope entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyLlama31RopePosIdsModule11:
    """FlashInfer apply_llama31_rope_pos_ids entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyRopeWithCosSinCacheModule12:
    """FlashInfer apply_rope_with_cos_sin_cache entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class ApplyRopeWithCosSinCacheInplaceModule13:
    """FlashInfer apply_rope_with_cos_sin_cache_inplace entry."""

    @func
    def run(
        q: Tensor[(BATCH * QUERY, Q_HEADS, HEAD_DIM), "bf16"],
        k: Tensor[(BATCH * QUERY, KV_HEADS, HEAD_DIM), "bf16"],
        cos: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        sin: ConstTensor[(BATCH * QUERY, HEAD_DIM // 2), "f32"],
        positions: Tensor[(BATCH * QUERY,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH * QUERY, 4, 32),
            names=("token", "warp", "lane"),
        ) as mesh:
            q_reg = tf.reshard(
                q, ((BATCH * QUERY) @ mesh.token, Q_HEADS @ mesh.warp, HEAD_DIM @ mesh.lane), "rmem"
            )
            k_smem = tf.reshard(
                k, ((BATCH * QUERY) @ mesh.token, KV_HEADS, HEAD_DIM @ mesh.lane), "smem"
            )
            cos_smem = tf.reshard(
                cos, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            sin_smem = tf.reshard(
                sin, ((BATCH * QUERY) @ mesh.token, (HEAD_DIM // 2) @ mesh.lane), "smem"
            )
            pos_reg = tf.reshard(positions, ((BATCH * QUERY) @ mesh.token,), "rmem")
            return tf.rope_with_positions(q_reg, k_smem, cos_smem, sin_smem, pos_reg)
