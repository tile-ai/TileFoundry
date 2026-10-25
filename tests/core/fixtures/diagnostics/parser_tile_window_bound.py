"""A tile loop window used as an explicit slice bound."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="stage", target=CpuTarget())
class TileWindowBound:
    @func
    def stage(x: Tensor[(1, 256, 64), "f32"]):
        out = x[:, 0:128, :]
        for t in tf.tile(128, 128):
            out = x[:, t : t + 128, :]
        return out
