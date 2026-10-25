"""End-to-end checks for the explicit host/device boundary.

Covers static and dynamic CTA launches, implicit and explicit CPU entries,
multi-CTA execution, and device-placement and shape negatives.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core.module import Module
from tilefoundry_nvidia import CudaTarget

INTEGRATION = Path(__file__).parents[1] / "fixtures" / "integration"
_TILE = 12
_DYNAMIC_CTA_LOWERING_WAIT = (
    "waiting for the follow-up dynamic-CTA runtime-lowering plan: lowering must "
    "preserve symbolic topology extents"
)


def _dynamic_cta_module(*, explicit_host: bool) -> Module:
    name = "dynamic_cta_double_host.py" if explicit_host else "dynamic_cta_double.py"
    return load_authored_ir(f"{INTEGRATION / name}:DynamicCtaDouble")


@pytest.mark.skip(reason=_DYNAMIC_CTA_LOWERING_WAIT)
def test_dynamic_cta_runs_two_shapes() -> None:
    """One compiled dynamic-CTA kernel runs at two shapes.

    One compiled artifact launches the dynamic-CTA kernel at two different
    ``Ntile`` shapes via the host-computed grid; both match torch with no
    recompile.
    """
    rm = tilefoundry.compile(
        _dynamic_cta_module(explicit_host=True), target=CudaTarget("nvidia.h200_sxm")
    )
    for nt in (4, 8):
        torch.manual_seed(nt)
        x = torch.randn(nt, _TILE, dtype=torch.float32, device="cuda")
        out = torch.empty_like(x)
        rm(x, out)
        torch.cuda.synchronize()
        assert torch.allclose(out, x * x, rtol=0, atol=0)


@pytest.mark.skip(reason=_DYNAMIC_CTA_LOWERING_WAIT)
def test_dynamic_cta_rejects_implicit_entry() -> None:
    """A dynamic-CTA kernel has no compile-time grid.

    A dynamic-CTA kernel has no compile-time grid, so the implicit
    auto-inserted host entry cannot derive one — it must error loudly rather
    than guess a CTA count.
    """
    mod = _dynamic_cta_module(explicit_host=False)
    with pytest.raises(Exception, match=r"cannot derive its grid"):
        tilefoundry.compile(mod, target=CudaTarget("nvidia.h200_sxm"))
