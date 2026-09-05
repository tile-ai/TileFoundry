from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types.shard import ComposedLayout, Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="sync_square_host")
class SyncSquare:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def sync_square_device(a: Tensor[(4, 32), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1)), names=('w', 't')) as m:
            view = T.tensor_view(a, layout=ShardLayout(layout=Layout(shape=(4, 32), strides=(32, 1)), attrs=(S(0), S(1)), mesh=Mesh(topologies=(Topology(name="thread", size=128),), layout=Layout(shape=(4, 32), strides=(32, 1)), names=("w", "t"))))
            reg = T.alloc_tensor(tensor_type=Tensor[(4, 32), "f32",
    ShardLayout(
        layout=Layout((4, 32), (32, 1)),
        attrs=(S(0), S(1)),
        mesh=Mesh((Topology("thread", 128),), Layout((4, 32), (32, 1)), names=('w', 't')),
    ), "rmem"])
            T.copy(view, reg)
            T.sync(m)
            T.sync(Mesh(topologies=(Topology(name="thread", size=128),), layout=ComposedLayout(inner=None, offset=0, outer=Layout(shape=(1, 32), strides=(32, 1))), names=("w", "t")))
            T.sync(Mesh(topologies=(Topology(name="thread", size=128),), layout=ComposedLayout(inner=None, offset=0, outer=Layout(shape=(2, 32), strides=(32, 1))), names=("w", "t")))
            T.sync(Mesh(topologies=(Topology(name="thread", size=128),), layout=ComposedLayout(inner=None, offset=64, outer=Layout(shape=(2, 32), strides=(32, 1))), names=("w", "t")))
            T.binary(reg, reg, reg, kind=BinaryKind.MUL)
            T.copy(reg, view)

    @prim_func(target=CpuTarget())
    def sync_square_host(a: Tensor[(4, 32), "f32"]):
        launch(sync_square_device, a, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821
