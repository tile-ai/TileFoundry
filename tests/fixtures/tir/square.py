from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import DimVar, T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.types.shard import Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.target import CpuTarget, CudaTarget

_S = DimVar("S", 1, 256)


@module(entry="square_host")
class TirSquare:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def square_device(x: Tensor[(_S,), "f32"]):
        pass

    @square_device.specialize(DimVarRangePat("S", 1, 127))
    def square_small(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(x, layout=ShardLayout(layout=Layout(shape=(128,), strides=(1,)), attrs=(S(0),), mesh=Mesh(topologies=(Topology(name="thread", size=128),), layout=Layout(shape=(128,), strides=(1,)), names=())))
            reg = T.alloc_tensor(tensor_type=Tensor[(128,), "f32",
                ShardLayout(
                    layout=Layout((128,), (1,)),
                    attrs=(S(0),),
                    mesh=Mesh((Topology("thread", 128),), Layout((128,), (1,))),
                ), "rmem"])
            for phase in range(0, 2, 1):
                if phase < 1:
                    T.copy(view, reg)
                else:
                    T.binary(reg, reg, reg, kind=BinaryKind.MUL)
                    T.copy(reg, view)
            T.sync(thread)

    @square_device.specialize(DimVarRangePat("S", 128, 255))
    def square_large(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(x, layout=ShardLayout(layout=Layout(shape=(128,), strides=(1,)), attrs=(S(0),), mesh=Mesh(topologies=(Topology(name="thread", size=128),), layout=Layout(shape=(128,), strides=(1,)), names=())))
            reg = T.alloc_tensor(tensor_type=Tensor[(128,), "f32",
                ShardLayout(
                    layout=Layout((128,), (1,)),
                    attrs=(S(0),),
                    mesh=Mesh((Topology("thread", 128),), Layout((128,), (1,))),
                ), "rmem"])
            for phase in range(0, 2, 1):
                if phase < 1:
                    T.copy(view, reg)
                else:
                    T.binary(reg, reg, reg, kind=BinaryKind.MUL)
                    T.copy(reg, view)
            T.sync(thread)

    @prim_func(target=CpuTarget())
    def square_host(x: Tensor[(_S,), "f32"]):
        launch(square_device, x, grid=(1, 1, 1), block=(128, 1, 1))  # noqa: F821
