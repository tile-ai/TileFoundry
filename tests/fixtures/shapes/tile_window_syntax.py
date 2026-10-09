"""Authored tile-window syntax shared by parser and printer tests."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types import Topology
from tilefoundry.target import CudaTarget


@func
def scan_copy(x: Tensor[(4, 4), "f32"]):
    out = tf.zeros(Tensor[(4, 4), "f32"])
    for row in tile(4, 2):
        out = tf.insert_slice(out, x[row, :], (row, 0))
    return out


@func
def nested_scan_copy(x: Tensor[(4, 4), "f32"]):
    out = tf.zeros(Tensor[(4, 4), "f32"])
    for row in tile(4, 2):
        for col in tile(4, 2):
            out = tf.insert_slice(out, x[row, col], (row, col))
    return out


_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = (Topology("cta", 1),)


@module(entry="windows", target=_H200, topologies=_CTA)
class NestedScaledTileWindows:
    @func
    def windows(x: Tensor[(10, 6), "f32"]):
        out = x[1:5:2, 0:2]
        for m in tile(1, 5, 2):
            for n in tile(1, 5, 2):
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
        for m in tile(1, 5, 2):
            with Mesh(("cta",), layout=(1,), names=("unit",)) as mesh:
                local = tf.reshard(x[m * 2 + 1, :], (2 @ mesh.unit, 8), "smem")
            out = local
        return out
