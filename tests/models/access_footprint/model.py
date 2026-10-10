"""Real matrix shapes whose authored loops expose reuse to analysis."""

from __future__ import annotations

from tests.fixtures.placed.flash_split_k_decode import FlashSplitKDecode
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
_CTAS = (Topology("cta", 132),)

S = 512
K = 2048
N = 4096
BM = 128
BN = 128
BK = 64

EXPERTS = 2
TOKENS = 120
HIDDEN = 64
MOE_BM = 10


@module(entry="qkv_projection", target=_H200, topologies=_CTAS)
class TiledQKVProjection:
    """The QKV projection shape tiled the way a kernel author writes it."""

    @func
    def qkv_projection(
        x: Tensor[(S, K), "bf16"],
        weight: ConstTensor[(K, N), "bf16"],
        output: Tensor[(S, N), "f32"],
    ) -> Tensor[(S, N), "f32"]:
        with Mesh(("cta",), layout=(132,), names=("cta",)) as _cta:
            result = output
            for m in tf.tile(S, BM):  # noqa: F405
                for n in tf.tile(N, BN):  # noqa: F405
                    acc = tf.zeros(
                        Tensor[(BM, BN), "f32", ((BM, BN), {}), "rmem"]
                    )
                    for k in tf.tile(K, BK):  # noqa: F405
                        lhs = tf.reshard(x[m, k], ((BM, BK), {}), "smem")
                        rhs = tf.reshard(weight[k, n], ((BK, BN), {}), "smem")
                        product = tf.matmul(lhs, rhs, out_dtype="f32")
                        acc = acc + tf.reshard(product, ((BM, BN), {}), "rmem")
                    result = tf.insert_slice(
                        result,
                        tf.reshard(acc, ((BM, BN), {}), "gmem"),
                        (m, n),
                    )
            return result


@module(entry="grouped_gemm", target=_H200, topologies=_CTAS)
class GroupedMoEGEMM:
    """Two experts consuming the token and hidden shapes of the MoE fixture."""

    @func
    def grouped_gemm(
        tokens: Tensor[(EXPERTS, TOKENS, HIDDEN), "f32"],
        weights: ConstTensor[(EXPERTS, HIDDEN, HIDDEN), "f32"],
        output: Tensor[(EXPERTS, TOKENS, HIDDEN), "f32"],
    ) -> Tensor[(EXPERTS, TOKENS, HIDDEN), "f32"]:
        with Mesh(("cta",), layout=(132,), names=("cta",)) as _cta:
            result = output
            for expert in tf.tile(EXPERTS, 1):  # noqa: F405
                for m in tf.tile(TOKENS, MOE_BM):  # noqa: F405
                    lhs = tf.reshard(
                        tf.reshape(
                            tokens[expert, m, 0:HIDDEN], (MOE_BM, HIDDEN)
                        ),
                        ((MOE_BM, HIDDEN), {}),
                        "smem",
                    )
                    rhs = tf.reshard(
                        tf.reshape(weights[expert, 0:HIDDEN, 0:HIDDEN], (HIDDEN, HIDDEN)),
                        ((HIDDEN, HIDDEN), {}),
                        "smem",
                    )
                    product = tf.matmul(
                        tf.cast(lhs, dtype="bf16"), tf.cast(rhs, dtype="bf16"), out_dtype="f32"
                    )
                    result = tf.insert_slice(
                        result,
                        tf.reshape(
                            tf.reshard(product, ((MOE_BM, HIDDEN), {}), "gmem"),
                            (1, MOE_BM, HIDDEN),
                        ),
                        (expert, m, 0),
                    )
            return result


__all__ = ["FlashSplitKDecode", "GroupedMoEGEMM", "TiledQKVProjection"]
