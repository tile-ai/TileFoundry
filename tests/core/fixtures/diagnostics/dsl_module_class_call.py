"""A Module class body that declares ``__call__``."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="only", target=CpuTarget())
class _OwnCall:
    @func
    def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)

    def __call__(self, x):
        raise AssertionError("never reached")
