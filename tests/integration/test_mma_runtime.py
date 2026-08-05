"""GPU end-to-end for ``tir.cuda.nn.Mma``.

Single SM80 16x8x16 BF16 mma atom: ``c = a @ b`` with ``a`` shape
(M=16, K=16) bf16, ``b`` shape (K=16, N=8) bf16, ``c`` shape
(M=16, N=8) f32. Numerical match against ``torch.matmul(a.float(),
b.float())`` within bf16 tolerance.
"""
from __future__ import annotations

import torch

import tilefoundry
from tilefoundry import func, module
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    ShardLayout,
    Split,
    Topology,
)

# ── Fragment layouts (cf. tests/ir_types/shard/test_mma_fragment_layouts.py) ──


_THREAD = Topology("thread", 32)
_THREAD_MESH = Mesh(
    topology=_THREAD,
    layout=Layout(shape=(4, 8), strides=(1, 4)),
    names=("x", "y"),
)

_A_FRAG = ShardLayout(
    layout=Layout(shape=(2, 4, 2, 8, 2), strides=(1, 2, 8, 16, 128)),
    attrs=(Split(1), Split(3)),
    mesh=_THREAD_MESH,
)
_B_FRAG = ShardLayout(
    layout=Layout(shape=(8, 2, 4, 2), strides=(1, 8, 16, 64)),
    attrs=(Split(2), Split(0)),
    mesh=_THREAD_MESH,
)
_C_FRAG = ShardLayout(
    layout=Layout(shape=(2, 4, 8, 2), strides=(1, 2, 8, 64)),
    attrs=(Split(1), Split(2)),
    mesh=_THREAD_MESH,
)


# ── @func DSL kernel: single-atom matmul (16,16) bf16 × (16,8) bf16 → (16,8) f32 ──


@module(entry="matmul_16x8x16")
class MatmulModule:
    @func
    def matmul_16x8x16(
        a: Tensor[(16, 16), "bf16"],
        b: Tensor[(16, 8), "bf16"],
    ) -> Tensor[(16, 8), "f32"]:
        # ShardLayouts carry the closure-captured ``_THREAD_MESH`` directly;
        # ``HirToTirPass`` derives the matching ``MeshScope`` from the body.
        a_frag = reshard(a, layout=_A_FRAG, storage="rmem")
        b_frag = reshard(b, layout=_B_FRAG, storage="rmem")
        c_frag = mma_sm80_16x8x16(
            a_frag, b_frag,
            dtype_a="bf16", dtype_b="bf16", dtype_acc="f32",
        )
        return reshard(c_frag, layout=_C_FRAG, storage="gmem")


def test_mma_sm80_16x8x16_bf16_matches_torch_matmul() -> None:
    rm = tilefoundry.compile(MatmulModule, target="cuda")
    a = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    rm(a, b, out)
    torch.cuda.synchronize()

    expected = torch.matmul(a.float(), b.float())
    # bf16 tolerance: each accumulated mac may have ≤ 2^-7 relative error,
    # over K=16 → cumulative ~ K · 2^-7 ≈ 0.125 in worst case.
    assert torch.allclose(out, expected, rtol=2e-2, atol=2e-2)
