"""The sync corpus: the case base and helpers its category modules share.

``tests/nvidia/cuda/analysis/test_sync.py`` imports each category module, so its cases
register as subclasses of ``SyncCase``.
"""

from __future__ import annotations

from pathlib import Path

from tests.core._corpus import Case
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.ir.core import get_metadata
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry_nvidia.cuda.analysis.sync import SyncMetadata, SyncPlanMetadata

FIXTURES = Path(__file__).parents[2] / "fixtures"
GEMM = FIXTURES / "schedule/hir/sm89_e4m3_gemm.py"


class SyncCase(Case):
    """A program the sync family plans, and the steps it places."""


def _attached(function):
    """Each node of *function* with the steps the family attached to it, in program order."""
    assert get_metadata(function, SyncPlanMetadata) is not None
    attached = {
        id(expr): (expr, record.steps)
        for expr in collect_exprs(function.body)
        if (record := get_metadata(expr, SyncMetadata)) is not None
    }
    return list(attached.values())


def _analyzed(module, analysis=("memory", "sync")):
    return analyze(*check_program(module, module.entry_function()), analysis=analysis)


def _planned(module):
    """The analysed entry Function, and each of its nodes with the steps attached to it."""
    function = _analyzed(module).function
    return function, _attached(function)
