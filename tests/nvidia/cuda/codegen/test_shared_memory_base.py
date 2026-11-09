"""The dynamic shared-memory base, declared once where every scope of a kernel reaches it."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tests.nvidia.cuda.ops.tir.targets import CUDA_TARGETS, retarget
from tilefoundry import cli
from tilefoundry_nvidia import cuda

CODEGEN = Path(__file__).parents[1] / "fixtures" / "codegen"


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_second_loop_reaches_the_shared_memory_base(target: cuda.Target) -> None:
    """Both loops stage ``x`` through offset 0, the second outside the block of the first."""
    TwoLoopStage = cli.load_authored_ir(f"{CODEGEN / 'two_loop_stage.py'}:TwoLoopStage")
    runtime = tilefoundry.compile(retarget(TwoLoopStage, target), target=target)
    source = torch.arange(8, dtype=torch.bfloat16, device="cuda").view(2, 4)
    first = torch.full_like(source, -1.0)
    second = torch.full_like(source, -1.0)
    runtime(source, first, second)
    torch.cuda.synchronize()
    assert torch.equal(first, source)
    assert torch.equal(second, source)
