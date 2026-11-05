"""Where the sync family orders shared memory and register accesses, and what it refuses.

The cases live in ``sync_cases``, one module per category.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import tilefoundry_nvidia.cuda.analysis.sync as sync
from tests.core._corpus import check_case, special
from tests.nvidia.cuda.analysis.sync_cases import (  # noqa: F401 -- register the cases
    GEMM,
    SyncCase,
    _planned,
    calls,
    cp_async,
    lowering,
    wgmma,
)
from tilefoundry.analysis import AnalysisError
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.tir.sync import SyncBarrier


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("within_scope", lambda mesh, scope: False, "is issued on a mesh not within its scope's"),
        ("classify", lambda mesh: SyncBarrier.GRID, "a scope that no thread mesh binds"),
        ("_LOOP_LIMIT", 1, "did not settle within its limit"),
    ],
    ids=["an issuing mesh outside its scope", "a barrier no thread mesh holds", "a loop that does not settle"],
)
def test_the_family_refuses_what_it_cannot_order(name, value, message):
    with patch.object(sync, name, value), pytest.raises(AnalysisError, match=message):
        _planned(load_authored_ir(str(GEMM)))


CASES = special(SyncCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_sync_case(case: SyncCase) -> None:
    check_case(case)
