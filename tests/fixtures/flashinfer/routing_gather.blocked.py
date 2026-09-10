"""FlashInfer M01 routing needs a batch-aware gather surface.

Notes:
upstream: flashinfer-ai/flashinfer @ 2ab910c58fdd2392914ea05e2a8714946ac0eef6
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: load
error: runtime_expression: unsupported call 'tf.gather' (2 positional, keywords ['axis', 'batch_dims'])
ledger: EXT-01
The three golden level variants preserve the routing equations; ``tf.gather``
with ``batch_dims=1`` has no current authored-HIR surface.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

ROWS, HIDDEN, K = 128, 16, 8
DT = "bf16"
TARGET = CudaTarget("nvidia.h200_sxm")
TOPOLOGIES = (Topology("cta", 8), Topology("thread", 128))


@module(entry="fused", target=TARGET, topologies=TOPOLOGIES)
class RoutingProgram:
    @func
    def score(x: Tensor[(ROWS, HIDDEN), DT], bias: ConstTensor[(HIDDEN,), DT]):
        return tf.sigmoid(x) + bias

    @func
    def unfused(x: Tensor[(ROWS, HIDDEN), DT], bias: ConstTensor[(HIDDEN,), DT]):
        s = score(x, bias)
        _, indices = tf.topk(s, k=K, axis=-1)
        return tf.gather(s, indices, axis=-1, batch_dims=1)

    @func
    def fused(x: Tensor[(ROWS, HIDDEN), DT], bias: ConstTensor[(HIDDEN,), DT]):
        s = tf.sigmoid(x) + bias
        _, indices = tf.topk(s, k=K, axis=-1, largest=True, sorted=True)
        return tf.gather(s, indices, axis=-1, batch_dims=1)


@module(entry="fused", target=TARGET, topologies=TOPOLOGIES)
class RoutingCTA:
    @func
    def score(x: Tensor[(ROWS, HIDDEN), DT], bias: ConstTensor[(HIDDEN,), DT]):
        return tf.sigmoid(x) + bias

    @func
    def unfused(x: Tensor[(ROWS, HIDDEN), DT], bias: ConstTensor[(HIDDEN,), DT]):
        s = score(x, bias)
        _, indices = tf.topk(s, k=K, axis=-1)
        return tf.gather(s, indices, axis=-1, batch_dims=1)

    @func
    def fused(x: Tensor[(ROWS, HIDDEN), DT], bias: ConstTensor[(HIDDEN,), DT]):
        s = tf.sigmoid(x) + bias
        _, indices = tf.topk(s, k=K, axis=-1, largest=True, sorted=True)
        return tf.gather(s, indices, axis=-1, batch_dims=1)


@module(entry="fused", target=TARGET, topologies=TOPOLOGIES)
class RoutingThread:
    @func
    def score(x: Tensor[(ROWS, HIDDEN), DT], bias: ConstTensor[(HIDDEN,), DT]):
        return tf.sigmoid(x) + bias

    @func
    def unfused(x: Tensor[(ROWS, HIDDEN), DT], bias: ConstTensor[(HIDDEN,), DT]):
        s = score(x, bias)
        _, indices = tf.topk(s, k=K, axis=-1)
        return tf.gather(s, indices, axis=-1, batch_dims=1)

    @func
    def fused(x: Tensor[(ROWS, HIDDEN), DT], bias: ConstTensor[(HIDDEN,), DT]):
        s = tf.sigmoid(x) + bias
        _, indices = tf.topk(s, k=K, axis=-1, largest=True, sorted=True)
        return tf.gather(s, indices, axis=-1, batch_dims=1)
