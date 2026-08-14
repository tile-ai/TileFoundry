"""Tile-window programs shared by analysis, parsing, and lowering tests."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import DimVar, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

WINDOW_SEQ = DimVar("seq", 4, 64)
WINDOW_TILE = DimVar("tile_size", 2, 8)


@func
def tile_window_add(
    x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]
) -> Tensor[(4, 4), "f32"]:
    out = tf.add(seed, seed)
    for row in tile(10, 4):
        out = tf.add(x[row, :], seed)
    return out


@func
def moved_tile_window_add(
    x: Tensor[(12, 4), "f32"], seed: Tensor[(3, 4), "f32"]
) -> Tensor[(3, 4), "f32"]:
    out = tf.add(seed, seed)
    for row in tile(6, 3):
        out = tf.add(x[row + 6, :], seed)
    return out


@func
def dynamic_tile_window_add(
    x: Tensor[(WINDOW_SEQ, 4), "f32"], seed: Tensor[(4, 4), "f32"]
) -> Tensor[(4, 4), "f32"]:
    out = tf.add(seed, seed)
    for row in tile(WINDOW_SEQ, 4):
        out = tf.add(x[row, :], seed)
    return out


@func
def unspecialized_tile_window_add(
    x: Tensor[(10, 4), "f32"], seed: Tensor[(WINDOW_TILE, 4), "f32"]
) -> Tensor[(WINDOW_TILE, 4), "f32"]:
    out = tf.add(seed, seed)
    for row in tile(10, WINDOW_TILE):
        out = tf.add(x[row, :], seed)
    return out


@module(
    entry="main",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class WindowCost:
    @func
    def main(source: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
        out = tf.add(seed, seed)
        for row in tile(10, 4):
            out = tf.add(source[row, :], seed)
        return out
