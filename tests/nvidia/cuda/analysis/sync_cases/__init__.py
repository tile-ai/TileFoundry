"""The sync corpus: the case base and helpers its category modules share.

``tests/nvidia/cuda/analysis/test_sync.py`` imports each category module, so its cases
register as subclasses of ``SyncCase``.
"""

from __future__ import annotations

from pathlib import Path

from tests.core._corpus import Case
from tilefoundry import analysis as tilefoundry_analysis
from tilefoundry.ir import core, expr
from tilefoundry_nvidia.cuda import analysis as cuda_analysis

FIXTURES = Path(__file__).parents[2] / "fixtures"
GEMM = FIXTURES / "schedule/hir/sm89_e4m3_gemm.py"


class SyncCase(Case):
    """A program the sync family plans, and the steps it places."""


def _attached(function):
    """Each node of *function* with the steps the family attached to it, in program order."""
    assert expr.get_metadata(function, cuda_analysis.SyncPlanMetadata) is not None
    attached = {
        id(node): (node, record.steps)
        for node in core.collect_exprs(function.body)
        if (record := expr.get_metadata(node, cuda_analysis.SyncMetadata)) is not None
    }
    return list(attached.values())


def _analyzed(module, analysis=("memory", "sync")):
    return tilefoundry_analysis.analyze(*tilefoundry_analysis.check_program(module, module.entry_function()), analysis=analysis)


def _planned(module):
    """The analysed entry Function, and each of its nodes with the steps attached to it."""
    function = _analyzed(module).function
    return function, _attached(function)
