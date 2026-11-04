"""A nested saved-state tuple passed between participant-local functions."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,), (1,)), ("dp",))
shard = ShardLayout(Layout((4, 2)), (Split(0),), mesh)

@module(entry="run", target=CudaTarget("nvidia.h200_sxm", device_count=2), topologies=(gpu,))
class TupleCall:
    @func
    def child(pair: tuple[Tensor[(4, 2), "f32", shard], tuple[Tensor[(4, 2), "f32", shard]]]):
        return pair[0] + pair[1][0]

    @func
    def run(x: Tensor[(4, 2), "f32", shard]):
        with Mesh(("gpu",), (2,), names=("dp",)) as _dp:
            return child((x, (x * 2.0,)))  # noqa: F821
