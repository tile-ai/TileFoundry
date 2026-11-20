"""Two pipeline stages, each on one slice of the mesh, joined by a Reshard over the whole mesh."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

target = cpu.Target()
gpu = Topology("gpu", 2)


@module(entry="forward", target=target, topologies=(gpu,))
class Pipeline:
    @func(mesh=Mesh(("gpu",), (2,), names=("pp",)))
    def forward(
        x: Tensor[(4, 8), "f32", (4, 8) @ mesh[0]],
        w0: Tensor[(8, 8), "f32", (8, 8) @ mesh[0]],
        w1: Tensor[(8, 8), "f32", (8, 8) @ mesh[1]],
    ):
        with Mesh(("gpu",), layout=(2,), names=("pp",)) as stages:
            with Mesh(stages[0], layout=(1,), names=("s",)) as _s0:
                h = tf.matmul(x, w0)
            moved = tf.reshard(h, (4, 8) @ stages[1], "gmem")
            with Mesh(stages[1], layout=(1,), names=("s",)) as _s1:
                return tf.matmul(moved, w1)
