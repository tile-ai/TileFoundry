"""Async-copy and barrier TIR programs shared by codegen and runtime tests."""

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import B, Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget


@module(entry="async_stage_host")
class AsyncStage:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def async_stage_device(
        a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]
    ):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,)), ("t",)) as m:
            a_view = T.tensor_view(
                a,
                layout=ShardLayout(Layout((128, 4), (4, 1)), (S(0),), m),
            )
            shared = T.alloc_tensor(
                TensorType(
                    (512,),
                    DType.f32,
                    ShardLayout(Layout((512,), (1,)), (B(),), m),
                    StorageKind.SMEM,
                )
            )
            T.copy_async(a_view, shared)
            T.cp_async_commit()
            T.cp_async_wait(n=0)
            T.sync(m)
            T.copy(shared, b)

    @prim_func(target=CpuTarget())
    def async_stage_host(
        a: Tensor[(128, 4), "f32"], b: Tensor[(128, 4), "f32"]
    ):
        launch(
            async_stage_device,
            a,
            b,
            grid=(1, 1, 1),
            block=(128, 1, 1),
        )  # noqa: F821


@module(entry="sync_square_host")
class SyncSquare:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def sync_square_device(a: Tensor[(4, 32), "f32"]):
        with Mesh(
            (Topology("thread", 128),), Layout((4, 32), (32, 1)), ("w", "t")
        ) as m:
            layout = ShardLayout(Layout((4, 32), (32, 1)), (S(0), S(1)), m)
            view = T.tensor_view(a, layout=layout)
            reg = T.alloc_tensor(
                TensorType((4, 32), DType.f32, layout, StorageKind.RMEM)
            )
            T.copy(view, reg)
            T.sync(m)
            T.sync(m[0, :])
            T.sync(m[0:2, :])
            T.sync(m[2:4, :])
            T.binary(reg, reg, reg, kind=BinaryKind.MUL)
            T.copy(reg, view)

    @prim_func(target=CpuTarget())
    def sync_square_host(a: Tensor[(4, 32), "f32"]):
        launch(
            sync_square_device,
            a,
            grid=(1, 1, 1),
            block=(128, 1, 1),
        )  # noqa: F821
