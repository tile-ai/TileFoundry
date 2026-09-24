from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="stride_host", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4), Topology("thread", 128),))
class TirGridStride:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def stride_device(x: Tensor[(256,), "f32"]):
        with Mesh((Topology("cta", 4),), Layout((4,), (1,)), names=('d0',)) as cta:
            with Mesh((Topology("thread", 128),), Layout((128,), (1,)), names=('t',)) as thread:
                view = T.tensor_view(x[0:0 + 128], layout=((128 @ thread.t,), (1,)))
                reg = T.alloc_tensor(tensor_type=Tensor[(128,), "f32", ((128 @ thread.t,), (1,)), "rmem"])
                for block in range(cta.d0, 16, 4):
                    T.copy(view, reg)
                    T.binary(reg, reg, reg, kind=BinaryKind.MUL)
                    T.copy(reg, view)
                T.sync(thread)

    @prim_func(target=CpuTarget())
    def stride_host(x: Tensor[(256,), "f32"]):
        launch(stride_device, x, grid=(4, 1, 1), block=(128, 1, 1))  # noqa: F821
