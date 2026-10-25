"""A tile loop window divided with true division."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="stage", target=CpuTarget())
class TileWindowDivision:
    @func
    def stage(x: Tensor[(1, 256, 64), "f32"]):
        out = x[:, 0:64, :]
        for t in tf.tile(128, 128):
            out = x[:, t / 2, :]
        return out
