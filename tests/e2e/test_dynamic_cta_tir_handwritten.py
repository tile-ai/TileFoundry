"""GPU e2e for a hand-written TIR dynamic-extent CTA kernel.

A hand-authored ``@prim_func`` whose CTA count is launch-provided: the device
mesh is ``Mesh(Topology("cta", None), Layout(shape=(None,), strides=(1,)))``
and the global tensor's leading dim is a ``DimVar``. Each CTA squares its own
``(1, TILE)`` row in place. One compiled artifact runs at several ``Ntile``
shapes with no recompile (the grid comes from the runtime tensor shape).

Mirrors the HIR reference ``dyn_double`` in ``test_host_launch.py`` but authored
entirely in TIR. Exercises the parser-side hidden ``<param>_shape_<axis>``
scalar injection that lets a hand-written device kernel read the runtime extent.
"""
from __future__ import annotations

import torch

import tilefoundry
from tilefoundry import module, prim_func
from tilefoundry.dsl import DimVar, T, Tensor
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology

_TILE = 12
_NT = DimVar("Ntile", 1, 64)


@module(entry="dyn_square_host")
class DynSquare:
    @prim_func(target="cuda")
    def dyn_square(a: Tensor[(_NT, _TILE), "f32"]):
        # Launch-provided CTA extent (None) → grid from the host launch; each CTA
        # owns one (1, TILE) row of the DimVar-shaped global tensor.
        with Mesh(Topology("cta", None), Layout(shape=(None,), strides=(1,))) as cta:
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

    @prim_func(target="cpu")
    def dyn_square_host(a: Tensor[(_NT, _TILE), "f32"]):
        launch(dyn_square, a, grid=(_NT, 1, 1), block=(1, 1, 1))  # noqa: F821


def test_handwritten_tir_dynamic_cta_matches_torch_at_several_shapes() -> None:
    """One compiled artifact squares the tensor at three ``Ntile`` shapes via
    the host-computed grid; all match torch with no recompile."""
    rm = tilefoundry.compile(DynSquare, target="cuda")
    for nt in (4, 8, 17):
        torch.manual_seed(nt)
        x = torch.randn(nt, _TILE, dtype=torch.float32, device="cuda")
        expected = x * x
        rm(x)
        torch.cuda.synchronize()
        assert torch.allclose(x, expected, rtol=0, atol=0)


def test_handwritten_tir_dynamic_cta_lowers_to_program_dim() -> None:
    """The dynamic CTA extent lowers to the runtime ``program_dim<cta>()`` path:
    no constexpr ``program_shape<cta>`` is emitted, and the runtime global dim
    flows through the hidden ``a_shape_0`` scalar."""
    from tilefoundry.codegen.cuda.module import emit_cuda_module  # noqa: PLC0415
    from tilefoundry.codegen.registry import group_functions_by_target  # noqa: PLC0415

    lowered = tilefoundry.lower(DynSquare, target="cuda")
    src = emit_cuda_module(group_functions_by_target(lowered)["cuda"]).source

    assert "tilefoundry::program_dim<tilefoundry::TopologyScope::cta>()" in src
    assert "program_shape<tilefoundry::TopologyScope::cta>" not in src
    assert "a_shape_0" in src
