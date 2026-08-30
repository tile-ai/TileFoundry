"""Mesh-coordinate slice starts used by installed diagnostic smoke tests."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

D, BLK, W, N = 64, 8, 4, 32
_H200, _CTA = CudaTarget("nvidia.h200_sxm"), Topology("cta", W)


@module(entry="scan", target=_H200, topologies=(_CTA,))
class Fixed:
    @func
    def scan(x: Tensor[(1, N, D), "bf16"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            acc = tf.full_like(tf.zeros(Tensor[(W @ m.w, 1, D), "f32", "smem"]), value=0.0)
            for t in tile(N, BLK * W):
                b0 = t
                blk = tf.reshard(x[:, b0 : b0 + BLK, :], (1, BLK, D), "smem")
                acc = acc + tf.cast(
                    tf.reduce(blk, axes=(1,), keepdim=True, kind="sum"), dtype="f32"
                )
            ga = tf.reshard(acc, (W, 1, D), "smem")
            return tf.reshape(
                tf.reshard(
                    tf.reduce(ga, axes=(0,), keepdim=False, kind="sum"),
                    (1, D),
                    "gmem",
                ),
                new_shape=(1, D),
            )


@module(entry="scan", target=_H200, topologies=(_CTA,))
class Strided:
    @func
    def scan(x: Tensor[(1, N, D), "bf16"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            acc = tf.full_like(tf.zeros(Tensor[(W @ m.w, 1, D), "f32", "smem"]), value=0.0)
            for t in tile(N, BLK * W):
                b0 = t + m.w * BLK
                blk = tf.reshard(x[:, b0 : b0 + BLK, :], (1, BLK, D), "smem")
                acc = acc + tf.cast(
                    tf.reduce(blk, axes=(1,), keepdim=True, kind="sum"), dtype="f32"
                )
            ga = tf.reshard(acc, (W, 1, D), "smem")
            return tf.reshape(
                tf.reshard(
                    tf.reduce(ga, axes=(0,), keepdim=False, kind="sum"),
                    (1, D),
                    "gmem",
                ),
                new_shape=(1, D),
            )


@module(entry="oob", target=_H200, topologies=(_CTA,))
class OutOfWindow:
    @func
    def oob(x: Tensor[(1, N, D), "bf16"]) -> Tensor[(1, BLK, D), "bf16"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as _mesh:
            bad = x[:, N : N + BLK, :]
            return tf.reshard(bad, (1, BLK, D), "gmem")
