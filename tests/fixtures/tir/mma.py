"""Hand-authored TIR MMA program shared by printer and runtime tests."""

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget

_OP = T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN


@module(entry="mm_host")
class MmHandwritten:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def mm_device(
        a: Tensor[(16, 16), "bf16"],
        b: Tensor[(16, 8), "bf16"],
        c: Tensor[(16, 8), "f32"],
    ):
        atom = T.cuda.mma.atom(op=_OP)
        with Mesh((Topology("thread", 32),), Layout((4, 8), (1, 4))) as _warp:
            a_view = T.tensor_view(a, layout=atom.A)
            b_view = T.tensor_view(b, layout=atom.B)
            a_frag = T.alloc_tensor(
                TensorType((16, 16), DType.bf16, atom.A, StorageKind.RMEM)
            )
            b_frag = T.alloc_tensor(
                TensorType((16, 8), DType.bf16, atom.B, StorageKind.RMEM)
            )
            acc = T.alloc_tensor(
                TensorType((16, 8), DType.f32, atom.C, StorageKind.RMEM)
            )
            T.copy(a_view, a_frag)
            T.copy(b_view, b_frag)
            T.fill(acc, 0.0)
            T.mma(acc, a_frag, b_frag, atom=atom)
            c_view = T.tensor_view(c, layout=atom.C)
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def mm_host(
        a: Tensor[(16, 16), "bf16"],
        b: Tensor[(16, 8), "bf16"],
        c: Tensor[(16, 8), "f32"],
    ):
        launch(mm_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821
