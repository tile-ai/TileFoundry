"""GPU e2e for the hand-written TIR MMA surface.

A hand-authored ``@prim_func`` SM80 16x8x16 matmul: platform-namespaced
`T.cuda.mma` op/atom, explicit `with Mesh((4,8),(1,4))`, register fragments via
`T.alloc_tensor(layout=atom.A/B/C)` filled by `T.copy` (distinct load / mma /
store ops), and `T.mma(acc, a, b, atom=atom)`. Compiled and run on GPU, matched
against ``torch.matmul`` within bf16 tolerance.

Mirrors the HIR reference in ``test_mma_runtime.py`` (same data + tolerance) but
authored entirely in TIR rather than lowered from HIR.
"""
from __future__ import annotations

import torch

import tilefoundry
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, Topology

_OP = T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN


@module(entry="mm_host")
class MmHandwritten:
    @prim_func(target="cuda")
    def mm_device(
        a: Tensor[(16, 16), "bf16"],
        b: Tensor[(16, 8), "bf16"],
        c: Tensor[(16, 8), "f32"],
    ):
        atom = T.cuda.mma.atom(op=_OP)
        with Mesh(Topology("thread", 32), Layout(shape=(4, 8), strides=(1, 4))) as warp:  # noqa: F841
            # gmem → register fragments (distributed load, like HIR reshard).
            a_view = T.tensor_view(a, layout=atom.A)
            b_view = T.tensor_view(b, layout=atom.B)
            a_frag = T.alloc_tensor(
                TensorType(shape=(16, 16), dtype=DType.bf16, layout=atom.A, storage=StorageKind.RMEM)
            )
            b_frag = T.alloc_tensor(
                TensorType(shape=(16, 8), dtype=DType.bf16, layout=atom.B, storage=StorageKind.RMEM)
            )
            acc = T.alloc_tensor(
                TensorType(shape=(16, 8), dtype=DType.f32, layout=atom.C, storage=StorageKind.RMEM)
            )
            T.copy(a_view, a_frag)
            T.copy(b_view, b_frag)
            T.fill(acc, 0.0)
            T.mma(acc, a_frag, b_frag, atom=atom)
            # register → gmem store.
            c_view = T.tensor_view(c, layout=atom.C)
            T.copy(acc, c_view)

    @prim_func(target="cpu")
    def mm_host(
        a: Tensor[(16, 16), "bf16"],
        b: Tensor[(16, 8), "bf16"],
        c: Tensor[(16, 8), "f32"],
    ):
        launch(mm_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821


def test_handwritten_tir_mma_matches_torch_matmul() -> None:
    rm = tilefoundry.compile(MmHandwritten, target="cuda")
    a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    rm(a, b, out)
    torch.cuda.synchronize()

    expected = torch.matmul(a.float(), b.float())
    assert torch.allclose(out, expected, rtol=2e-2, atol=2e-2)
