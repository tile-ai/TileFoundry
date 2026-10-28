"""End-to-end authored TIR control-flow runtime coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import tvm_ffi

import tilefoundry
from tests.core._corpus import Case, case, check_case, special
from tilefoundry_nvidia import CudaTarget


class ControlFlowCase(Case):
    def guard(self, state, problems):
        torch.manual_seed(0)
        x = torch.randn(128, dtype=torch.float32, device="cuda")
        expected = x.square()
        runtime = tilefoundry.compile(state, target=CudaTarget("nvidia.h200_sxm"))
        assert runtime.name == f"{state.name}.square_host"
        runtime(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(x, expected, rtol=0, atol=0)


@case
class Square(ControlFlowCase):
    path: Path = Path(__file__).parents[1] / "fixtures/tir/square.py"
    name: str | None = "TirSquare"


@case
class MultipleHosts(ControlFlowCase):
    path: Path = Path(__file__).parents[1] / "fixtures/tir/square_multiple_hosts.py"
    name: str | None = "TirSquareMultipleHosts"

    def guard(self, state, problems):
        super().guard(state, problems)
        second = torch.randn(128, dtype=torch.float32, device="cuda")
        expected = second.square()
        tvm_ffi.get_global_func(f"{state.name}.secondary_host")(second)
        torch.cuda.synchronize()
        torch.testing.assert_close(second, expected, rtol=0, atol=0)


CASES = special(ControlFlowCase)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_for_if_square_matches_torch(case) -> None:
    check_case(case)
