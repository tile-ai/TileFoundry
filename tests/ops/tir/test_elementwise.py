"""A stride-0 operand of ``ops::elementwise``, which is the whole of broadcast.

See [runtime §3](docs/spec/runtime.md#3-runtime-ops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tests.fixtures.tir.layouts import bcast
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.target import CpuTarget, CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="col_bcast_host")
class ColumnBroadcast:
    """``dst(m, k) = lhs(m, k) * rhs(m)``, with ``rhs`` a stride-0 column."""

    @prim_func(target=_CUDA)
    def col_bcast_device(
        a: Tensor[(32,), "f32"], r: Tensor[(4,), "f32"], out: Tensor[(32,), "f32"]
    ):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as m:
            a_view = T.tensor_view(a, layout=bcast((32,), (1,), m))
            r_view = T.tensor_view(r, layout=bcast((4,), (1,), m))
            out_view = T.tensor_view(out, layout=bcast((32,), (1,), m))
            lhs = T.alloc_tensor(
                Tensor[(4, 8), 'f32', bcast((4, 8), (8, 1), m), 'smem']
            )
            col = T.alloc_tensor(
                Tensor[(4, 1), 'f32', bcast((4, 1), (1, 1), m), 'smem']
            )
            dst = T.alloc_tensor(
                Tensor[(4, 8), 'f32', bcast((4, 8), (8, 1), m), 'smem']
            )
            T.copy(a_view, lhs)
            T.copy(r_view, col)
            T.sync(m)
            T.binary(lhs, col, dst, kind=BinaryKind.MUL)
            T.copy(dst, out_view)

    @prim_func(target=CpuTarget())
    def col_bcast_host(a: Tensor[(32,), "f32"], r: Tensor[(4,), "f32"], out: Tensor[(32,), "f32"]):
        launch(col_bcast_device, a, r, out, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_column_is_reread_for_every_element_of_its_row() -> None:
    """Each of the 8 elements of row ``m`` multiplies by the one ``r[m]``."""
    rm = tilefoundry.compile(ColumnBroadcast, target=_CUDA)
    torch.manual_seed(0)
    a = torch.randn(32, dtype=torch.float32, device="cuda")
    r = torch.randn(4, dtype=torch.float32, device="cuda")
    out = torch.zeros(32, dtype=torch.float32, device="cuda")
    rm(a, r, out)
    torch.cuda.synchronize()
    expected = a * r.repeat(8)
    assert torch.allclose(out, expected, rtol=0, atol=0)
