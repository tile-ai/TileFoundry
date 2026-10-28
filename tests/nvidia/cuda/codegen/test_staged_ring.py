"""The staged-ring schedule fixtures, compiled and run on the card.

``staged_dynamic_inner`` and ``staged_nested`` reach full CUDA emission
through bound tuples of shared-memory views and bound runtime stage choices.
Their rings are
re-authored with what execution needs and the corpus does not state --
the dynamic-smem carve-out, the cp.async commit/wait before read-back,
and trip counts uniform across the warp.

See [runtime §2.6.2](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#262-opscopycuh).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
import tilefoundry_nvidia.cuda.codegen  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry.cli.source import load_authored_ir
from tilefoundry_nvidia import CudaTarget

CODEGEN = Path(__file__).parents[1] / "fixtures" / "codegen"
_CUDA = CudaTarget("nvidia.h200_sxm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_counter_ring_copies_tile() -> None:
    """The runtime index picks a stage per iteration; the tile arrives intact.

    A choice rendered wrong -- an index that does not read the counter, or
    stage bases spelled so the subscript cannot compile -- either fails
    emission here outright or lands the tile nowhere near ``out``.
    """
    CounterRing = load_authored_ir(f"{CODEGEN / 'counter_ring.py'}:CounterRing")
    rm = tilefoundry.compile(CounterRing, target=_CUDA)
    x = torch.arange(8, dtype=torch.bfloat16, device="cuda").view(1, 8)
    out = torch.full((1, 8), -1.0, dtype=torch.bfloat16, device="cuda")
    rm(x, out)
    torch.cuda.synchronize()
    assert torch.equal(out, x), (
        "counter-driven ring: iteration k copies x through stage k % 2, so "
        "out must hold x exactly -- a stale or uninitialized stage read "
        "breaks the equality"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_row_ring_copies_rows() -> None:
    """Dim arithmetic over both loop counters drives the ring; rows arrive.

    The index is ``((i * 3) + j) % 2`` rendered for the card, over windows
    whose bases come out of the slice path, so a row landed anywhere but its
    slot's stage breaks the row-for-row equality.
    """
    RowRing = load_authored_ir(f"{CODEGEN / 'row_ring.py'}:RowRing")
    rm = tilefoundry.compile(RowRing, target=_CUDA)
    x = torch.arange(48, dtype=torch.bfloat16, device="cuda").view(6, 8)
    out = torch.full((6, 8), -1.0, dtype=torch.bfloat16, device="cuda")
    rm(x, out)
    torch.cuda.synchronize()
    assert torch.equal(out, x), (
        "row ring: row 3i + j passes through stage (3i + j) % 2 into out's "
        "row 3i + j, so out must equal x row for row"
    )
