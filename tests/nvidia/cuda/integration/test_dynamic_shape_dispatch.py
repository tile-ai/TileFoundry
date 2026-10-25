"""GPU end-to-end for dynamic-shape dispatch.

A ``pass`` prototype partitions ``DimVar('S', 1, 7)`` into closed ranges:
``[1, 4)`` squares and ``[4, 8)`` doubles. One kernel holds both arms and
picks between them on the extent it was told, so one binary handles every
shape in the envelope.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry.cli.source import load_authored_ir
from tilefoundry_nvidia import CudaTarget

INTEGRATION = Path(__file__).parents[1] / "fixtures" / "integration"


def _build_runtime_module():
    """Compile the dispatch prototype to a fully-loaded RuntimeModule."""
    Dispatch = load_authored_ir(f"{INTEGRATION / 'shape_range_dispatch.py'}:Dispatch")
    return tilefoundry.compile(Dispatch, target=CudaTarget("nvidia.h200_sxm"))


_HIR_LOWERING_ELSEWHERE = (
    "this program is authored in HIR, and HIR reaches TIR outside this pipeline; "
    "the claim it makes about dispatch is pinned on TIR in "
    "tests/nvidia/cuda/codegen/test_device_dispatch.py"
)


@pytest.mark.skip(reason=_HIR_LOWERING_ELSEWHERE)
def test_entry_dispatch_both_variants_in_one_session() -> None:
    """Both dispatch arms run through the same compiled binary in sequence.

    Both dispatch arms run through the same compiled binary in
    sequence. Verifies that the dispatch entry routes each call to its
    matching variant based on the current tensor's runtime shape.
    """
    rm = _build_runtime_module()

    x_a = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
    out_a = torch.empty_like(x_a)
    rm(x_a, out_a)

    x_b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float32, device="cuda")
    out_b = torch.empty_like(x_b)
    rm(x_b, out_b)
    torch.cuda.synchronize()

    assert torch.allclose(out_a, x_a * x_a, rtol=0, atol=0)
    assert torch.allclose(out_b, x_b + x_b, rtol=0, atol=0)
