"""End-to-end compilation of child CUDA Modules with independent topologies."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry.cli.source import load_authored_ir
from tilefoundry_nvidia import CudaTarget

CODEGEN = Path(__file__).parents[1] / "fixtures" / "codegen"
_CUDA = CudaTarget("nvidia.h200_sxm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_parent_links_two_topology_domains() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(32, device="cuda")
    x1 = torch.randn(128, device="cuda")
    o0 = torch.empty_like(x0)
    o1 = torch.empty_like(x1)
    _Parent = load_authored_ir(f"{CODEGEN / 'two_topology_domains.py'}:_Parent")
    runtime = tilefoundry.compile(_Parent, target=_CUDA)
    runtime(x0, o0, x1, o1)
    torch.cuda.synchronize()
    torch.testing.assert_close(o0, x0)
    torch.testing.assert_close(o1, x1)
