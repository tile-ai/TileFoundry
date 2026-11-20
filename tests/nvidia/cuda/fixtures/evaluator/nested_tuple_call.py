"""A nested saved-state tuple passed between unit-local functions."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

gpu = Topology("gpu", 2)

@module(entry="run", target=cuda.Target("nvidia.h200_sxm", device_count=2), topologies=(gpu,))
class TupleCall:
    mesh = Mesh(("gpu",), (2,), names=("dp",))

    @func
    def child(pair: Tuple[Tensor[(4, 2), "f32", ((4 @ mesh.dp, 2), (2, 1))], Tuple[Tensor[(4, 2), "f32", ((4 @ mesh.dp, 2), (2, 1))]]]):
        return pair[0] + pair[1][0]

    @func
    def run(x: Tensor[(4, 2), "f32", ((4 @ mesh.dp, 2), (2, 1))]):
        with Mesh(("gpu",), (2,), names=("dp",)) as _dp:
            return child((x, (x * 2.0,)))
