"""Authored tile-window syntax shared by parser and printer tests."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = (Topology("cta", 1),)


@module(entry="windows", target=_H200, topologies=_CTA)
class NestedScaledTileWindows:
    @func
    def windows(x: Tensor[(10, 6), "f32"]):
        out = x[1:5:2, 0:2]
        for m in tf.tile(1, 5, 2):
            for n in tf.tile(1, 5, 2):
                out = (
                    x[m * 2, n]
                    + x[2 * m, n - 1]
                    + x[m * 2 + 1, n]
                    + x[m * 2 - 1, n]
                )
        return out


@module(entry="window", target=_H200, topologies=_CTA)
class MeshInsideTileWindow:
    @func
    def window(
        x: Tensor[(10, 8), "f32"], seed: Tensor[(2, 8), "f32", "smem"]
    ):
        out = seed
        for m in tf.tile(1, 5, 2):
            with Mesh(("cta",), layout=(1,), names=("unit",)) as mesh:
                local = tf.reshard(x[m * 2 + 1, :], (2 @ mesh.unit, 8), "smem")
                out = tf.reshard(local, (2, 8), "smem")
        return out
