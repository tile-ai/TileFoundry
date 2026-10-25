"""Cover the fused multiply-contract: what it refuses, and the one call it emits.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
import tilefoundry_nvidia.cuda.codegen  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.tir.dot import Dot
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Return, Sequential
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_nvidia import CudaTarget

OPS = Path(__file__).parents[2] / "fixtures" / "ops"


def _pf(*types) -> PrimFunction:
    args = tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    return PrimFunction(
        target=CudaTarget("nvidia.h200_sxm"),
        name="fn",
        params=args,
        body=Sequential(
            body=(
                MeshScope(
                    mesh=_MESH,
                    binding=Var(name="thread", type=_MESH),
                    body=Sequential(body=(Evaluate(callable=Dot(), args=args),)),
                ),
                Return(),
            )
        ),
    )


def _ty(n, dtype=DType.from_name("f32"), storage="rmem"):
    return make_tensor_type((n,), dtype, storage=storage)


_MESH = Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",))


def _sharded(n, storage="rmem"):
    """An operand the block-contraction check accepts, so the workspace is what is tested."""
    return make_tensor_type(
        (n,),
        DType.from_name("f32"),
        storage=storage,
        layout=ShardLayout(Layout((n,), (1,)), (Broadcast(),), _MESH),
    )


def test_dot_accepts_equal_runs() -> None:
    verify_prim_function(_pf(_ty(8), _ty(8), _ty(1)))


def test_dot_accepts_mixed_dtypes() -> None:
    """The fold accumulates in f32 whatever it loads, so the two may differ."""
    verify_prim_function(_pf(_ty(8, DType.from_name("bf16")), _ty(8), _ty(1)))


def test_dot_refuses_length_mismatch() -> None:
    """The fold walks one operand's length and indexes the other with it."""
    with pytest.raises(VerifyError, match="contract over different lengths"):
        verify_prim_function(_pf(_ty(8), _ty(4), _ty(1)))


def test_dot_refuses_wide_destination() -> None:
    """A contraction leaves a total, and a total is one number."""
    with pytest.raises(VerifyError, match="must be one cell"):
        verify_prim_function(_pf(_ty(8), _ty(8), _ty(4)))


def test_dot_refuses_gmem_workspace() -> None:
    """One warp posts its partial and every thread reads the posted ones."""
    with pytest.raises(VerifyError, match=r"workspace: gmem does not match smem"):
        verify_prim_function(
            _pf(_sharded(8), _sharded(8), _sharded(1), _sharded(4, storage="gmem"))
        )


def test_dot_refuses_unsharded_block_lhs() -> None:
    """The warps that post a partial are the ones lhs's mesh names."""
    with pytest.raises(VerifyError, match="must carry a ShardLayout"):
        verify_prim_function(_pf(_ty(8), _ty(8), _ty(1), _ty(4, storage="smem")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dot_tiers_match_torch() -> None:
    """One compile behind two assertions, each naming the tier that failed."""
    DotTiers = load_authored_ir(f"{OPS / 'dot_tiers.py'}:DotTiers")
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
