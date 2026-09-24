"""A window of a tile whose modes are grouped by the axis they belong to.

A staged box writes each axis as the modes that reach it rather than as one
extent and one stride, so a window of it has no single stride per axis to
walk. What it reaches is the modes the axis steps the least by, as many of
them as its size takes. The placement sugar states one extent per axis and
so has no spelling for such a box; it is written out in full instead.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.ir.types.shard import Layout, Topology
from tilefoundry.target import CudaTarget

ROWS, COLS, BM, BN = 16, 64, 8, 16
_BOX = Layout(((2, 8), (4, 16)), ((512, 64), (16, 1)))
_H200 = CudaTarget("nvidia.h200_sxm")


@module(entry="corner", target=_H200, topologies=(Topology("cta", 1),))
class GroupedWindow:
    @func
    def corner(x: Tensor[(ROWS, COLS), "f32", _BOX, "smem"]) -> Tensor[(BM, BN), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("c",)) as _c:
            return tf.reshard(x[0:BM, 0:BN], (BM, BN), "gmem")
