"""Each way a type names a mesh slice, and a scope opened on a slice without a name.

``@ mesh[0]`` is Broadcast on the units of the first ``pp`` row; ``mesh[0].pp @ V()``
is Varying there; ``8 @ mesh[1].cp`` splits over the second row, and a swizzled
layout is parenthesized before ``@``. ``@ s0`` names a
mesh bound by ``with`` that has no parent; ``with stages[1]:`` opens a slice unnamed.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="run", target=cpu.Target(), topologies=(Topology("gpu", 4),))
class SliceLayouts:
    @func(mesh=Mesh(("gpu",), (2, 2), names=("pp", "cp")))
    def run(
        a: Tensor[(4, 8), "f32", (4, 8) @ mesh[0]],
        b: Tensor[(4, 8), "f32", (4, 8) @ {mesh[0].pp @ V()}],
        c: Tensor[(8, 16), "f32", (8 @ mesh[1].cp, 16)],
        d: Tensor[(8, 8), "f32", (Layout((8, 8), (8, 1)) | Swizzle(1, 2, 3)) @ mesh[1]],
    ):
        with Mesh(("gpu",), layout=(2, 2), names=("pp", "cp")) as stages:
            with Mesh(stages[0], layout=(2,), names=("s",)) as s0:
                h = tf.reshard(a + b, (4, 8) @ s0, "gmem")
            moved = tf.reshard(h, (4, 8) @ stages[1], "gmem")
            with stages[1]:
                return moved * 2.0
