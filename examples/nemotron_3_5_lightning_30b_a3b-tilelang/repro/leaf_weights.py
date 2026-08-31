#!/usr/bin/env python
"""Checking one leaf function draws every weight the module declares.

`leaf` takes two activations and no `ConstTensor` at all. `check` on it still
materialises `entry`'s weight, on the semantic side and on the runtime side, so
the memory a leaf check needs is the whole model's. On the real model -- 474
declared weights, about 60 GB -- that is 139 GB and there is no way to run it:

    $ tilefoundry check runtime_model.py:Nemotron35Lightning30BA3BRuntime.attend \\
          --inputs random --dim ctx_full=0 --dim ctx_tail=128 --out output --fn nan_inf
    CUDA out of memory. Tried to allocate 27.36 GiB. GPU 0 has a total capacity
    of 139.80 GiB of which 605.12 MiB is free.

Here the weight is a knob. At BIG_GB = 1 the leaf check takes a second; raise it
until the card says no, and note that what ran out was a check of a function
that declares nothing.

    $ BIG_GB=1  python -c "import repro.leaf_weights"    # then:
    $ tilefoundry check repro/leaf_weights.py:Mod --fn nan_inf --out output \\
          --inputs random --dim n=64
    $ tilefoundry check repro/leaf_weights.py:Mod.leaf --fn nan_inf --out output \\
          --inputs random --dim n=64
"""
from __future__ import annotations

import os

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, DimVar, Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

D, W = 64, 4
#: How big the weight nobody asked for is, in GiB.
BIG_GB = float(os.environ.get("BIG_GB", "1"))
BIG = int(BIG_GB * (1 << 30) / 2 / D)
N = DimVar("n", 1, 1024)
_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", W)


@module(entry="entry", target=_H200, topologies=(_CTA,))
class Mod:
    """One heavy function and one that declares no weight at all."""

    @func
    def leaf(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            return tf.reshard(xs + xs, (1, D), "gmem")

    @func
    def entry(x: Tensor[(1, D), "f32"], k: Tensor[(1, N), "f32"],
              big: ConstTensor[(BIG, D), "bf16"]) -> Tensor[(1, D), "f32"]:
        with Mesh(("cta",), layout=(W,), names=("w",)) as m:
            xs = tf.reshard(x, (1, D @ m.w), "smem")
            ws = tf.reshard(big[0:1, :], (1, D @ m.w), "smem")
            return tf.reshard(xs + tf.cast(ws, dtype="f32"), (1, D), "gmem")
