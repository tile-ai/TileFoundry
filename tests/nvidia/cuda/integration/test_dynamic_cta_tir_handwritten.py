"""GPU end-to-end for a hand-written dynamic-extent CTA kernel.

A launch-provided CTA mesh and leading ``DimVar`` let each CTA square one row.
One artifact runs several shapes without recompilation. This TIR twin of
``dyn_double`` pins parser injection of the hidden parameter-shape scalar used
to read the runtime extent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry.cli.source import load_authored_ir
from tilefoundry_nvidia import CudaTarget

INTEGRATION = Path(__file__).parents[1] / "fixtures" / "integration"
_TILE = 12
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
    DynSquare = load_authored_ir(f"{INTEGRATION / 'dynamic_cta_square.py'}:DynSquare")
    rm = tilefoundry.compile(DynSquare, target=CudaTarget("nvidia.h200_sxm"))
    for nt in (4, 8, 17):
        torch.manual_seed(nt)
        x = torch.randn(nt, _TILE, dtype=torch.float32, device="cuda")
        expected = x * x
        rm(x)
        torch.cuda.synchronize()
        assert torch.allclose(x, expected, rtol=0, atol=0)
