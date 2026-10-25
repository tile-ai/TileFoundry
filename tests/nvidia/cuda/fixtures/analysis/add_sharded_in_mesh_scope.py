"""An add sharded over the threads of a cta x thread scope."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 2), Topology("thread", 4))


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class WithScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        with Mesh(("cta",), (2,), ("tile",)) as _cta:
            with Mesh(("thread",), (4,), ("t",)) as thread:
                local = tf.reshard(x, (8, 16 @ thread.t), "rmem")
                return tf.reshard(local + local, (8, 16), "gmem")
