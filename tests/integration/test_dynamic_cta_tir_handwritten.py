"""GPU end-to-end for a hand-written dynamic-extent CTA kernel.

A launch-provided CTA mesh and leading ``DimVar`` let each CTA square one row.
One artifact runs several shapes without recompilation. This TIR twin of
``dyn_double`` pins parser injection of the hidden parameter-shape scalar used
to read the runtime extent.
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tilefoundry import module, prim_func
from tilefoundry.dsl import DimVar, T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget

_TILE = 12
_NT = DimVar("Ntile", 1, 64)
_DYNAMIC_CTA_LOWERING_WAIT = (
    "waiting for the follow-up dynamic-CTA runtime-lowering plan: lowering must "
    "preserve symbolic topology extents"
)


@pytest.mark.skip(reason=_DYNAMIC_CTA_LOWERING_WAIT)
def test_handwritten_tir_dynamic_cta_matches_torch_at_several_shapes() -> None:
    """Test handwritten tir dynamic cta matches torch at several shapes.

    One compiled artifact squares the tensor at three ``Ntile`` shapes via
    the host-computed grid; all match torch with no recompile.
    """
    @module(entry="dyn_square_host")
    class DynSquare:
        @prim_func(target=CudaTarget("nvidia.h200_sxm"))
        def dyn_square(a: Tensor[(_NT, _TILE), "f32"]):
            with Mesh((Topology("cta", _NT),), Layout(shape=(_NT,), strides=(1,))) as cta:
                a_view = T.tensor_view(
                    a,
                    layout=ShardLayout(
                        layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                        attrs=(Split(0),),
                        mesh=cta,
                    ),
                )
                reg = T.alloc_tensor(
                    TensorType(
                        shape=(_NT, _TILE),
                        dtype=DType.f32,
                        layout=ShardLayout(
                            layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                            attrs=(Split(0),),
                            mesh=cta,
                        ),
                        storage=StorageKind.RMEM,
                    )
                )
                T.copy(a_view, reg)
                T.binary(reg, reg, reg, kind=BinaryKind.MUL)
                T.copy(reg, a_view)

        @prim_func(target=CpuTarget())
        def dyn_square_host(a: Tensor[(_NT, _TILE), "f32"]):
            launch(dyn_square, a, grid=(_NT, 1, 1), block=(1, 1, 1))  # noqa: F821

    rm = tilefoundry.compile(DynSquare, target=CudaTarget("nvidia.h200_sxm"))
    for nt in (4, 8, 17):
        torch.manual_seed(nt)
        x = torch.randn(nt, _TILE, dtype=torch.float32, device="cuda")
        expected = x * x
        rm(x)
        torch.cuda.synchronize()
        assert torch.allclose(x, expected, rtol=0, atol=0)
