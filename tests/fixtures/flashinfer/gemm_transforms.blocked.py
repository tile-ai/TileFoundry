"""Expected authored HIR for G01, T01-T02, and L01.

Notes:
upstream: flashinfer-ai/flashinfer @ 2ab910c58fdd2392914ea05e2a8714946ac0eef6
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: DType: unknown value 'nvfp4'
ledger: EXT-04, OP-04, OP-14
This is negative conformance input using proposed authored-HIR operations.
"""

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Mesh, Topology
from tilefoundry.target import CudaTarget

TARGET = CudaTarget("nvidia.h200_sxm")
TOPOS = (Topology("cta", 8), Topology("thread", 128))
M, K, N, R = 128, 64, 128, 16


# noqa
@module(entry="fused_cta", target=TARGET, topologies=TOPOS)
class G01ResidualGemmLoraBias:
    @func
    def unfused(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(N, K), "nvfp4"],
        d: Tensor[(M, R), "bf16"],
        l1: Tensor[(N, R), "bf16"],
        bias: Tensor[(N,), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:
        base = tf.nvfp4_matmul(a, tf.transpose(b, perm=(1, 0)))
        delta = tf.matmul(d, tf.transpose(l1, perm=(1, 0)))
        return base + delta + bias

    @func
    def fused_program(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(N, K), "nvfp4"],
        d: Tensor[(M, R), "bf16"],
        l1: Tensor[(N, R), "bf16"],
        bias: Tensor[(N,), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:

        acc = tf.nvfp4_mma_accumulate(a, b, accumulator="f32")
        acc = tf.lora_up_accumulate(acc, d, l1)
        return tf.epilogue(acc, bias=bias, alpha=1.0, output_dtype="bf16")

    @func
    def fused_cta(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(N, K), "nvfp4"],
        d: Tensor[(M, R), "bf16"],
        l1: Tensor[(N, R), "bf16"],
        bias: Tensor[(N,), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(8,)) as cta:
            a_tile = tf.reshard(a, (M @ cta, K), "smem")
            d_tile = tf.reshard(d, (M @ cta, R), "smem")
            b_tile = tf.reshard(b, (N, K), "smem")
            l1_tile = tf.reshard(l1, (N, R), "smem")
            bias_tile = tf.reshard(bias, (N,), "smem")
            acc = tf.nvfp4_mma_accumulate(a_tile, b_tile, accumulator="rmem")
            acc = tf.lora_up_accumulate(acc, d_tile, l1_tile)
            out = tf.epilogue(acc, bias=bias_tile, alpha=1.0, output_dtype="bf16")
            return tf.reshard(out, (M, N), "gmem")

    @func
    def fused_thread(
        a: Tensor[(M, K), "bf16"],
        b: Tensor[(N, K), "nvfp4"],
        d: Tensor[(M, R), "bf16"],
        l1: Tensor[(N, R), "bf16"],
        bias: Tensor[(N,), "bf16"],
    ) -> Tensor[(M, N), "bf16"]:
        with Mesh(("thread",), layout=(4, 32), names=("warp", "lane")) as thread:
            acc = tf.mma_fragment(a, b, owner=thread, storage="rmem", input_format="nvfp4")
            acc = tf.lora_up_accumulate(acc, d, l1, owner=thread)
            out = tf.epilogue(acc, bias=bias, output_dtype="bf16")
            return tf.reshard(out, (M, N), "gmem")


# noqa
@module(entry="topk_page_cta", target=TARGET, topologies=TOPOS)
class TopKIndexTransforms:
    @func
    def topk_page_unfused(
        scores: Tensor[(M, N), "bf16"], page_table: Tensor[(M, N), "i32"]
    ) -> Tensor[(M, 8), "i32"]:
        _, indices = tf.topk(scores, k=8)
        return tf.page_table_transform(indices, page_table)

    @func
    def topk_page_cta(
        scores: Tensor[(M, N), "bf16"], page_table: Tensor[(M, N), "i32"]
    ) -> Tensor[(M, 8), "i32"]:
        with Mesh(("cta",), layout=(8,)) as cta:
            local = tf.reshard(scores, (M @ cta, N), "smem")
            _, indices = tf.topk(local, k=8)
            transformed = tf.page_table_transform(indices, page_table, storage="smem")
            return tf.reshard(transformed, (M, 8), "gmem")

    @func
    def topk_page_thread(
        scores: Tensor[(M, N), "bf16"], page_table: Tensor[(M, N), "i32"]
    ) -> Tensor[(M, 8), "i32"]:
        with Mesh(("thread",), layout=(128,)) as thread:
            local = tf.reshard(scores, (M @ thread, N), "rmem")
            _, indices = tf.topk(local, k=8)
            transformed = tf.page_table_transform(indices, page_table, storage="rmem")
            return tf.reshard(transformed, (M, 8), "gmem")

    @func
    def topk_ragged_cta(
        scores: Tensor[(M, N), "bf16"], indptr: Tensor[(M + 1,), "i32"]
    ) -> Tensor[(M, 8), "i32"]:
        with Mesh(("cta",), layout=(8,)) as cta:
            local = tf.reshard(scores, (M @ cta, N), "smem")
            _, indices = tf.topk(local, k=8)
            return tf.reshard(tf.ragged_index_transform(indices, indptr), (M, 8), "gmem")


# noqa
@module(entry="fused_thread", target=TARGET, topologies=TOPOS)
class LogitsProcessorChain:
    @func
    def unfused(
        logits: Tensor[(M, N), "f32"], temperature: Tensor[(M, 1), "f32"]
    ) -> Tensor[(M, N), "f32"]:
        scaled = logits / temperature
        penalized = tf.repetition_penalty(scaled)
        return tf.top_p_mask(tf.top_k_mask(penalized))

    @func
    def fused_thread(
        logits: Tensor[(M, N), "f32"], temperature: Tensor[(M, 1), "f32"]
    ) -> Tensor[(M, N), "f32"]:
        with Mesh(("thread",), layout=(128,)) as thread:
            local = tf.reshard(logits, (M @ thread, N), "rmem")
            scaled = local / temperature
            penalized = tf.repetition_penalty(scaled, storage="rmem")
            selected = tf.fused_top_k_top_p_mask(penalized, storage="rmem")
            return tf.reshard(selected, (M, N), "gmem")


# noqa
# noqa
# noqa
# noqa
