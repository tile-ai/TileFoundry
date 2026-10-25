"""Read the far half of a fused tensor through a window moved by a constant."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget

_HALF, _COLS, _STEP = 4, 4, 2


@module(entry="moved_copy", target=CpuTarget(), topologies=(Topology("thread", 1),))
class _MovedWindow:
    @func
    def moved_copy(gu: Tensor[(2 * _HALF, _COLS), "f32"]):
        with Mesh(("thread",), (1,), ("t",)) as m:
            gr = tf.reshard(gu, (2 * _HALF, _COLS @ m.t), "rmem")
            acc = tf.full_like(gr, 0.0)
            for r in tf.tile(_HALF, _STEP):
                acc = tf.insert_slice(acc, gr[r + _HALF, :], (r, 0))
            return tf.reshard(acc, (2 * _HALF, _COLS @ m.t), "gmem")
