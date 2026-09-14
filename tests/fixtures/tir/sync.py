from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="sync_square_host", target=CudaTarget("nvidia.h200_sxm"))
class SyncSquare:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def sync_square_device(a: Tensor[(4, 32), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1)), names=('w', 't')) as m:
            view = T.tensor_view(a, layout=((4 @ m.w, 32 @ m.t), (32, 1)))
            reg = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32", ((4 @ m.w, 32 @ m.t), (32, 1)), "rmem"])
            T.copy(view, reg)
            T.sync(m)
            T.sync(m[:1])
            T.sync(m[:2])
            T.sync(m[2:])
            T.binary(reg, reg, reg, kind=BinaryKind.MUL)
            T.copy(reg, view)

    @prim_func(target=CpuTarget())
    def sync_square_host(a: Tensor[(4, 32), "f32"]):
        launch(sync_square_device, a, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821
