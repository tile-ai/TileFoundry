"""Loop carries whose shard attr changes across one step, each way between Broadcast and Varying."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu, cta = Topology("gpu", 2), Topology("cta", 1)
mesh = Mesh((gpu,), Layout((2,)), ("g",))
ROW = TensorType((1, 8), DType.f32, ShardLayout.from_logical(
    (1, 8), mesh, (Broadcast(),), storage="gmem", owners=target.storage_owners), "gmem")
VARYING = TensorType((1, 8), DType.f32, ShardLayout.from_logical(
    (1, 8), mesh, (Varying(),), storage="gmem", owners=target.storage_owners), "gmem")


@module(entry="forward", target=target, topologies=(gpu, cta))
class EntersBroadcast:
    @func
    def forward(acc0: ROW, step: Tensor[(1, 8), "f32", "gmem"]):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _m:
            acc = acc0
            for _t in range(4):
                acc = acc + step
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class EntersVarying:
    @func
    def forward(acc0: VARYING, row: ROW):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _m:
            acc = acc0
            for _t in range(4):
                acc = row * 2.0
            return acc
