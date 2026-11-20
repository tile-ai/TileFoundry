"""Where the sync family orders shared memory and register accesses, and what it refuses.

The cases live in ``sync_cases``, one module per category.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.core._corpus import check_case, special
from tests.nvidia.cuda.analysis.sync_cases import (
    GEMM,
    SyncCase,
    _planned,
    calls,
    cp_async,
    lowering,
    wgmma,
)
from tilefoundry import analysis, cli
from tilefoundry.cli import schedule
from tilefoundry.ir import tir


@pytest.mark.parametrize(
    ("owner", "name", "value", "message"),
    [
        ("tilefoundry.ir.utils", "within_mesh_scope", lambda mesh, scope: False, "is issued on a mesh not within its scope's"),
        ("tilefoundry.ir.tir.sync", "classify", lambda mesh: tir.sync.SyncBarrier.GRID, "a scope that no thread mesh binds"),
        ("tilefoundry_nvidia.cuda.analysis.sync", "LOOP_LIMIT", 1, "did not settle within its limit"),
    ],
    ids=["an issuing mesh outside its scope", "a barrier no thread mesh holds", "a loop that does not settle"],
)
def test_the_family_refuses_what_it_cannot_order(owner, name, value, message):
    with patch(f"{owner}.{name}", value), pytest.raises(analysis.Error, match=message):
        _planned(cli.load_authored_ir(str(GEMM)))


CATEGORIES = (calls, cp_async, lowering, wgmma)
CASES = special(SyncCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_sync_case(case: SyncCase) -> None:
    check_case(case)


@pytest.mark.parametrize(("fixture", "module"), (("tma_store_cta_windows", "TmaStoreCtaWindows"), ("tma_store_cta_slice", "TmaStoreCtaSlice")))
def test_a_varying_smem_tile_is_still_shared_by_the_threads_of_its_cta(tmp_path, fixture, module):
    """A tile Varying on the CTA axis is one copy per CTA, which its threads share.

    Its layout names no thread topology, but the executing threads share the
    bytes, so the TMA load into it waits for the previous TMA store from it: the
    loop starts with a barrier, as for a plain tile.
    """
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / f"{fixture}.py"
    out = tmp_path / "lowered.py"
    schedule.run_finalize(f"{source}:{module}", str(out))
    assert out.read_text().count("T.sync(") == 2
