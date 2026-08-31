#!/usr/bin/env python
"""A slice whose start depends on the mesh index.

Splitting a long reduction over a worker axis is the ordinary shape for it: each
unit walks every W-th block, so the block it reads starts at `t + w * BLK`. The
evaluator cannot run that -- it reports

    shape '[]' is invalid for input of size 4

where 4 is the width of the mesh axis, with nothing to say which slice it was.
The same function with `b0 = t` (every unit reading the same block, which is not
the program anyone wants but is a legal one) runs.

    $ tilefoundry check repro/mesh_slice_start.py:Fixed  --inputs random \\
          --out output --fn nan_inf
    $ tilefoundry check repro/mesh_slice_start.py:Strided --inputs random \\
          --out output --fn nan_inf
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare tile()
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

D, BLK, W, N = 64, 8, 4, 32
_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", W)


@module(entry="scan", target=_H200, topologies=(_CTA,))
class Fixed:
    """Every unit reads the same block: the slice start is mesh-independent."""

    @func
    def scan(x: Tensor[(1, N, D), "bf16"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            a0 = tf.zeros(Tensor[(W @ m.w, 1, D), "f32", "smem"])
            acc = tf.full_like(a0, value=0.0)
            for t in tile(N, BLK * W):
                b0 = t
                blk = tf.reshard(x[:, b0:b0 + BLK, :], (1, BLK, D), "smem")
                acc = acc + tf.cast(tf.reduce(blk, axes=(1,), keepdim=True, kind="sum"),
                                    dtype="f32")
            ga = tf.reshard(acc, (W, 1, D), "smem")
            return tf.reshape(tf.reshard(tf.reduce(ga, axes=(0,), keepdim=False,
                                                   kind="sum"), (1, D), "gmem"),
                              new_shape=(1, D))


@module(entry="scan", target=_H200, topologies=(_CTA,))
class Strided:
    """Unit `w` reads block `w` of each group: the slice start carries `m.w`."""

    @func
    def scan(x: Tensor[(1, N, D), "bf16"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            a0 = tf.zeros(Tensor[(W @ m.w, 1, D), "f32", "smem"])
            acc = tf.full_like(a0, value=0.0)
            for t in tile(N, BLK * W):
                b0 = t + m.w * BLK
                blk = tf.reshard(x[:, b0:b0 + BLK, :], (1, BLK, D), "smem")
                acc = acc + tf.cast(tf.reduce(blk, axes=(1,), keepdim=True, kind="sum"),
                                    dtype="f32")
            ga = tf.reshard(acc, (W, 1, D), "smem")
            return tf.reshape(tf.reshard(tf.reduce(ga, axes=(0,), keepdim=False,
                                                   kind="sum"), (1, D), "gmem"),
                              new_shape=(1, D))
