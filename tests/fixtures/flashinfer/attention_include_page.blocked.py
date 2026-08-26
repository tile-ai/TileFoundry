"""Indexed paged-cache append with functional cache ownership.

upstream: flashinfer-ai/flashinfer@2ab910c58fdd2392914ea05e2a8714946ac0eef6 include/flashinfer/page.cuh
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: runtime_expression: unsupported call 'tf.scatter_update' (3 positional, keywords ['axis'])
ledger: OP-08

Each SURVEY entry remains an independent Module with the source group's
representative supported specialization and upstream mechanism intact.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

BATCH, QUERY, CONTEXT = 4, 128, 2048
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
TARGET = CudaTarget("nvidia.h200_sxm")
CTA_COUNT, THREAD_COUNT = BATCH * KV_HEADS, 4 * 32
TOPOLOGIES = (Topology("cta", CTA_COUNT), Topology("thread", THREAD_COUNT))


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class AppendPagedKVCacheDecodeKernelModule1:
    """FlashInfer AppendPagedKVCacheDecodeKernel entry."""

    @func
    def run(
        cache: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        values: Tensor[(BATCH, 1, KV_HEADS, HEAD_DIM), "bf16"],
        slots: Tensor[(BATCH,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, KV_HEADS, 4, 32),
            names=("batch", "kv_head", "warp", "lane"),
        ) as mesh:
            cache_smem = tf.reshard(
                cache,
                (
                    BATCH @ mesh.batch,
                    CONTEXT @ mesh.warp,
                    KV_HEADS @ mesh.kv_head,
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            values_reg = tf.reshard(
                values,
                (BATCH @ mesh.batch, 1, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "rmem",
            )
            slots_reg = tf.reshard(slots, (BATCH @ mesh.batch,), "rmem")
            return tf.scatter_update(cache_smem, slots_reg, values_reg, axis=1)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class AppendPagedKVCacheKernelModule2:
    """FlashInfer AppendPagedKVCacheKernel entry."""

    @func
    def run(
        cache: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        values: Tensor[(BATCH, 1, KV_HEADS, HEAD_DIM), "bf16"],
        slots: Tensor[(BATCH,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, KV_HEADS, 4, 32),
            names=("batch", "kv_head", "warp", "lane"),
        ) as mesh:
            cache_smem = tf.reshard(
                cache,
                (
                    BATCH @ mesh.batch,
                    CONTEXT @ mesh.warp,
                    KV_HEADS @ mesh.kv_head,
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            values_reg = tf.reshard(
                values,
                (BATCH @ mesh.batch, 1, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "rmem",
            )
            slots_reg = tf.reshard(slots, (BATCH @ mesh.batch,), "rmem")
            return tf.scatter_update(cache_smem, slots_reg, values_reg, axis=1)


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class NVFP4QuantizeAppendPagedKVCacheKernelModule3:
    """FlashInfer NVFP4QuantizeAppendPagedKVCacheKernel entry."""

    @func
    def run(x: Tensor[(BATCH, QUERY, Q_HEADS, HEAD_DIM), "bf16"]):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, Q_HEADS, QUERY // 16, 8, 16),
            names=("batch", "head", "token_block", "token", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (
                    BATCH @ mesh.batch,
                    QUERY @ (mesh.token_block, mesh.token),
                    Q_HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (
                    BATCH @ mesh.batch,
                    QUERY @ (mesh.token_block, mesh.token),
                    Q_HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            packed, scales = tf.quant(tf.cast(x_smem, "f32"), group=16, target_dtype="nvfp4")
            return packed, scales


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class NVFP4QuantizeAppendPagedKVCacheWithSlotMappingKernelModule4:
    """FlashInfer NVFP4QuantizeAppendPagedKVCacheWithSlotMappingKernel entry."""

    @func
    def run(x: Tensor[(BATCH, QUERY, Q_HEADS, HEAD_DIM), "bf16"]):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, Q_HEADS, QUERY // 16, 8, 16),
            names=("batch", "head", "token_block", "token", "lane"),
        ) as mesh:
            x_reg = tf.reshard(
                x,
                (
                    BATCH @ mesh.batch,
                    QUERY @ (mesh.token_block, mesh.token),
                    Q_HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "rmem",
            )
            x_smem = tf.reshard(
                x_reg,
                (
                    BATCH @ mesh.batch,
                    QUERY @ (mesh.token_block, mesh.token),
                    Q_HEADS @ mesh.head,
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            packed, scales = tf.quant(tf.cast(x_smem, "f32"), group=16, target_dtype="nvfp4")
            return packed, scales


@module(entry="run", target=TARGET, topologies=TOPOLOGIES)
class AppendPagedKVMlaCacheKernelModule5:
    """FlashInfer AppendPagedKVMlaCacheKernel entry."""

    @func
    def run(
        cache: Tensor[(BATCH, CONTEXT, KV_HEADS, HEAD_DIM), "bf16"],
        values: Tensor[(BATCH, 1, KV_HEADS, HEAD_DIM), "bf16"],
        slots: Tensor[(BATCH,), "i32"],
    ):
        with Mesh(
            ("cta", "thread"),
            layout=(BATCH, KV_HEADS, 4, 32),
            names=("batch", "kv_head", "warp", "lane"),
        ) as mesh:
            cache_smem = tf.reshard(
                cache,
                (
                    BATCH @ mesh.batch,
                    CONTEXT @ mesh.warp,
                    KV_HEADS @ mesh.kv_head,
                    HEAD_DIM @ mesh.lane,
                ),
                "smem",
            )
            values_reg = tf.reshard(
                values,
                (BATCH @ mesh.batch, 1, KV_HEADS @ mesh.kv_head, HEAD_DIM @ mesh.lane),
                "rmem",
            )
            slots_reg = tf.reshard(slots, (BATCH @ mesh.batch,), "rmem")
            return tf.scatter_update(cache_smem, slots_reg, values_reg, axis=1)
