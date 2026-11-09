"""Where the sync family orders shared memory and register accesses, and what it refuses.

The cases live in ``sync_cases``, one module per category.
"""

from __future__ import annotations

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
from tilefoundry.ir import tir


@pytest.mark.parametrize(
    ("owner", "name", "value", "message"),
    [
        ("tilefoundry.ir.types", "within_scope", lambda mesh, scope: False, "is issued on a mesh not within its scope's"),
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
