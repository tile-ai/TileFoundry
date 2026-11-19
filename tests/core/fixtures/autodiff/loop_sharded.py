"""A recurrence over a batch split on two GPUs: each stack keeps the split, one axis further in."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class LoopSharded:
    @func(mesh=Mesh(("gpu",), (2,), names=("data",)))
    def forward(h0: Tensor[(4 @ mesh.data, 3), "f32", "gmem"], w: Tensor[(3, 3), "f32", (3, 3) @ {B()}, "gmem"]):
        h = h0
        for _ in range(3):
            h = tf.matmul(h * h, w)
        return h
