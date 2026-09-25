"""Cover the fused multiply-contract: what it refuses, and the one call it emits.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.dot import Dot
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.tir.verify import verify_prim_function
from tilefoundry.ir.types import DType, Layout, Mesh, ShardLayout, Split, Topology, make_tensor_type
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.target import CpuTarget, CudaTarget


def _pf(*types) -> PrimFunction:
    args = tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    return PrimFunction(
        name="fn",
        params=args,
        body=Sequential(body=(Evaluate(callable=Dot(), args=args), Return())),
    )


def _ty(n, dtype=DType.from_name("f32"), storage="rmem"):
    return make_tensor_type((n,), dtype, storage=storage)


def test_accepts_two_equal_runs_folded_into_one_cell() -> None:
    verify_prim_function(_pf(_ty(8), _ty(8), _ty(1)))


def test_accepts_operands_of_different_element_types() -> None:
    """The fold accumulates in f32 whatever it loads, so the two may differ."""
    verify_prim_function(_pf(_ty(8, DType.from_name("bf16")), _ty(8), _ty(1)))


def test_refuses_operands_that_contract_over_different_lengths() -> None:
    """The fold walks one operand's length and indexes the other with it."""
    with pytest.raises(VerifyError, match="contract over different lengths"):
        verify_prim_function(_pf(_ty(8), _ty(4), _ty(1)))


def test_refuses_a_destination_wider_than_one_cell() -> None:
    """A contraction leaves a total, and a total is one number."""
    with pytest.raises(VerifyError, match="must be one cell"):
        verify_prim_function(_pf(_ty(8), _ty(8), _ty(4)))


def test_refuses_a_workspace_outside_shared_memory() -> None:
    """One warp posts its partial and every thread reads the posted ones."""
    with pytest.raises(VerifyError, match="workspace must be smem"):
        verify_prim_function(_pf(_ty(8), _ty(8), _ty(1), _ty(4, storage="gmem")))


def test_refuses_a_block_contraction_over_an_unsharded_left_operand() -> None:
    """The warps that post a partial are the ones lhs's mesh names."""
    with pytest.raises(VerifyError, match="must carry a ShardLayout"):
        verify_prim_function(_pf(_ty(8), _ty(8), _ty(1), _ty(4, storage="smem")))


@module(topologies=(Topology("thread", 32),))
class DotWarp:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def dot_warp(
        warp_a: Tensor[(32, 32), "f32"],
        warp_b: Tensor[(32,), "f32"],
        warp_c: Tensor[(32,), "f32"],
    ):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as mw:
            warp_a_view = T.tensor_view(
                warp_a,
                layout=ShardLayout(
                    layout=Layout(shape=(32, 32), strides=(32, 1)),
                    attrs=(Split(0),),
                    mesh=mw,
                ),
            )
            warp_b_view = T.tensor_view(
                warp_b,
                layout=ShardLayout(
                    layout=Layout(shape=(32,), strides=(1,)), attrs=(Broadcast(),), mesh=mw
                ),
            )
            warp_c_view = T.tensor_view(
                warp_c,
                layout=ShardLayout(
                    layout=Layout(shape=(32,), strides=(1,)), attrs=(Split(0),), mesh=mw
                ),
            )
            T.dot(warp_a_view, warp_b_view, warp_c_view)


@module(topologies=(Topology("thread", 128),))
class DotCta:
    @prim_func(target=CudaTarget("nvidia.h200_sxm"))
    def dot_cta(
        cta_a: Tensor[(128,), "f32"], cta_b: Tensor[(128,), "f32"], cta_c: Tensor[(1,), "f32"]
    ):
        with Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)), ("t",)) as mc:
            cta_a_view = T.tensor_view(cta_a, layout=ShardLayout(Layout((128,), (1,)), (Split(0),), mc))
            cta_b_view = T.tensor_view(cta_b, layout=ShardLayout(Layout((128,), (1,)), (Split(0),), mc))
            cta_c_view = T.tensor_view(cta_c, layout=ShardLayout(Layout((1,), (1,)), (Broadcast(),), mc))
            cta_ws = T.alloc_tensor(Tensor[(4,), "f32", None, "smem"])
            T.dot(cta_a_view, cta_b_view, cta_c_view, cta_ws)


@module(entry="dot_tiers_host", target=CudaTarget("nvidia.h200_sxm"))
class DotTiers:
    warp = DotWarp
    cta = DotCta

    @prim_func(target=CpuTarget())
    def dot_tiers_host(warp_a: Tensor[(32, 32), "f32"], warp_b: Tensor[(32,), "f32"], warp_c: Tensor[(32,), "f32"], cta_a: Tensor[(128,), "f32"], cta_b: Tensor[(128,), "f32"], cta_c: Tensor[(1,), "f32"]):
        launch(warp.dot_warp, warp_a, warp_b, warp_c, grid=(1, 1, 1), block=(32, 1, 1))
        launch(cta.dot_cta, cta_a, cta_b, cta_c, grid=(1, 1, 1), block=(128, 1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_both_tiers_answer_what_torch_answers() -> None:
    """One compile behind two assertions, each naming the tier that failed."""
    rm = tilefoundry.compile(DotTiers, target=CudaTarget("nvidia.h200_sxm"))
    torch.manual_seed(0)
    warp_a = torch.randn(32, 32, dtype=torch.float32, device="cuda")
    warp_b = torch.randn(32, dtype=torch.float32, device="cuda")
    warp_c = torch.zeros(32, dtype=torch.float32, device="cuda")
    cta_a = torch.randn(128, dtype=torch.float32, device="cuda")
    cta_b = torch.randn(128, dtype=torch.float32, device="cuda")
    cta_c = torch.zeros(1, dtype=torch.float32, device="cuda")
    rm(warp_a, warp_b, warp_c, cta_a, cta_b, cta_c)
    torch.cuda.synchronize()
    assert torch.allclose(warp_c, (warp_a @ warp_b).sum().expand(32), rtol=1e-4, atol=1e-4), (
        "warp tier (32-lane mesh, no workspace beside it): lane_axis_extent "
        "reads the mesh's fastest axis and this one is exactly 32, the only "
        "width the butterfly is. Each lane contracts its own row of a against "
        "the broadcast b and the butterfly leaves every lane holding the sum of "
        "all 32, so the reference is the whole matrix-vector product summed"
    )
    assert torch.allclose(cta_c, (cta_a * cta_b).sum().reshape(1), rtol=1e-4, atol=1e-4), (
        "block tier (workspace argument on a 128-thread mesh): four warps each "
        "post one partial and every thread folds all four, so the count of slots "
        "read comes off the operands' mesh (128 / 32) and not off blockDim"
    )
