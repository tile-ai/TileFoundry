"""Sync cases for calls whose callees touch shared memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import case
from tests.nvidia.cuda.analysis.sync_cases import FIXTURES, SyncCase, _planned
from tilefoundry import analysis

NESTED_CALLS = FIXTURES / "analysis/sync_nested_calls.py"


@case
class PlainNestedCalls(SyncCase):
    """A call whose callees hold no shared memory is admitted, and needs no step."""

    path: Path = NESTED_CALLS
    name: str | None = "Plain"

    def guard(self, state, problems):
        _function, attached = _planned(state)
        assert attached == []


@case
class StagedNestedCalls(SyncCase):
    """Once the innermost leaf holds shared memory, the call is one action touching any bytes.

    Nothing else in the root touches shared memory, so the call needs no step.
    """

    path: Path = NESTED_CALLS
    name: str | None = "Staged"

    def guard(self, state, problems):
        _function, attached = _planned(state)
        assert attached == []


@case
class TwiceNestedCalls(SyncCase):
    """Two calls whose callees touch shared memory need a barrier between them.

    The calls sit in the CTA region, where no thread mesh can hold that barrier.
    """

    path: Path = NESTED_CALLS
    name: str | None = "Twice"

    def guard(self, state, problems):
        with pytest.raises(analysis.Error, match="a scope that no thread mesh binds"):
            _planned(state)
