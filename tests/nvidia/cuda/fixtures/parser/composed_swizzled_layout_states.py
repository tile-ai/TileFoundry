"""Composed, swizzled layout states on a parameter and on a reshard."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="f", target=cuda.Target("nvidia.h200_sxm"))
class Model:
    @func(mesh=Mesh((Topology("thread", 8),), layout=(8,), names=("lane",)))
    def f(
        x: Tensor[
            (8, 16), "f32",
            ((8, 16) + 4 | Swizzle(2, 4, 3)) @ {mesh.lane @ P("sum")},
            "rmem",
        ],
    ):
        return tf.reshard(
            x, ((8, 16) + 4 | Swizzle(2, 4, 3)) @ {B()}, "smem"
        )
