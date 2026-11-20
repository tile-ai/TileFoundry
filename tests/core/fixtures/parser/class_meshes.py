"""Meshes declared once in the class body and named by signatures and bodies.

Pipeline is the two-stage program of the plan: ``pp`` is declared in the class, its
slices place each stage's values, and the body opens it with ``with pp:``. TwoMeshes
declares two meshes; three functions name them, and ``both`` takes one parameter on
each.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

gpu = Topology("gpu", 4)


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class Pipeline:
    pp = Mesh(("gpu",), (2,), names=("pp",))

    @func
    def forward(x: Tensor[(4, 8) @ pp[0], "f32"],
                w0: Tensor[(8, 8) @ pp[0], "f32"],
                w1: Tensor[(8, 8) @ pp[1], "f32"]):
        with pp:
            with pp[0]:
                h = tf.matmul(x, w0)
            moved = tf.reshard(h, (4, 8) @ pp[1], "gmem")
            with pp[1]:
                return tf.matmul(moved, w1)


@module(entry="both", target=cpu.Target(), topologies=(gpu,))
class TwoMeshes:
    rows = Mesh(("gpu",), (4,), names=("r",))
    grid = Mesh(("gpu",), (2, 2), names=("dp", "tp"))

    @func
    def by_rows(x: Tensor[(8, 4), "f32", (8 @ rows.r, 4)]):
        return x

    @func
    def by_grid(x: Tensor[(8, 4), "f32", (8 @ grid.dp, 4 @ grid.tp)]):
        return x

    @func
    def both(x: Tensor[(8, 4), "f32", (8 @ rows.r, 4)], y: Tensor[(8, 4), "f32", (8 @ grid.dp, 4 @ grid.tp)]):
        return x, y
