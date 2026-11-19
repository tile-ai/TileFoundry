"""Loop carries whose shard attr changes across one step, each way between Broadcast and Varying."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu, cta = Topology("gpu", 2), Topology("cta", 1)


@module(entry="forward", target=target, topologies=(gpu, cta))
class EntersBroadcast:
    @func(mesh=Mesh(("gpu",), (2,), names=("g",)))
    def forward(acc0: Tensor[(1, 8), "f32", (1, 8) @ {B()}, "gmem"], step: Tensor[(1, 8), "f32", "gmem"]):
        acc = acc0
        for _t in range(4):
            acc = acc + step
        return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class EntersVarying:
    @func(mesh=Mesh(("gpu",), (2,), names=("g",)))
    def forward(acc0: Tensor[(1, 8), "f32", "gmem"], row: Tensor[(1, 8), "f32", (1, 8) @ {B()}, "gmem"]):
        acc = acc0
        for _t in range(4):
            acc = row * 2.0
        return acc
