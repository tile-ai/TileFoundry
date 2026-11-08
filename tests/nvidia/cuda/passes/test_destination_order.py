"""Reads of the data a schedule overwrites in place are lowered before the overwrite.

See [analysis §1.2.2](docs/spec/analysis.md#122-memory).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.nvidia.cuda.schedule._support import _RetainedStatements
from tilefoundry.analysis import AnalysisError
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.liveness import LivenessMetadata
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import get_metadata
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.types import StorageKind
from tilefoundry.schedule import finalize

FIXTURE = Path(__file__).parents[1] / "fixtures/passes/destination_order.py"


def _analyzed(name: str):
    module = load_authored_ir(f"{FIXTURE}:{name}")
    return analyze(*check_program(module, module.entry_function()), analysis=("memory", "sync"))


def _lowered(result) -> list[str]:
    """The copies of an analysed program in TIR order: read from shared memory or written to it."""
    statements = _RetainedStatements()
    statements.visit(finalize(result).lookup(result.function.name).body)
    return [
        "read" if stmt.args[0].type.storage is StorageKind.SMEM else "write"
        for stmt in statements.statements
        if isinstance(stmt.callable, Copy)
    ]


@pytest.mark.parametrize(
    "name, order",
    (
        ("Sibling", ["read", "write"]),
        ("MaybeTrip", ["write", "read", "write", "read", "write", "read"]),
        ("CaptureHandoff", ["write", "write", "read", "read", "write", "read"]),
        ("KeptAcross", ["write", "read", "read", "write", "read", "read"]),
        ("CrossHalves", ["write", "read", "read", "write", "read"]),
    ),
    ids=(
        "sibling regions in a loop body",
        "inner loop that may run no rounds",
        "captured tile handed to a carry",
        "carry that copies the tile before the overwrite",
        "old half read beside a new half of one tile",
    ),
)
def test_old_data_is_read_before_it_is_overwritten(name, order):
    """Events and statements follow the order liveness records for each reordered scope."""
    result = _analyzed(name)
    liveness = get_metadata(result.function, LivenessMetadata)
    defined = {id(item.value.expr): item.defined_at for item in liveness.intervals}
    assert liveness.orders
    for children in liveness.orders.values():
        events = [defined[id(child)] for child in children if id(child) in defined]
        assert events == sorted(events)
    assert _lowered(result) == order


@pytest.mark.parametrize(
    "name",
    ("OneTrip", "DisjointHandoff", "WriteKept", "CarryBeforeLoad", "CarryOtherHalf"),
    ids=(
        "loop that always overwrites its carry",
        "loop handing on the half it does not overwrite",
        "load into a carried copy while the original is read after",
        "carried copy read while the body loads into the original",
        "carried half beside a load into the other half",
    ),
)
def test_reads_already_before_the_overwrite_keep_the_order(name):
    """A read that cannot touch what a write replaces, or already precedes it, keeps its order."""
    assert get_metadata(_analyzed(name).function, LivenessMetadata).orders == {}


@pytest.mark.parametrize(
    "name",
    ("YieldOld", "PartialReload", "ReturnOld"),
    ids=(
        "loop handing on the tile it overwrites",
        "reload of half a tile a full load took first",
        "result handing on the row it overwrites",
    ),
)
def test_an_overwrite_the_analysis_cannot_order_is_refused(name):
    with pytest.raises(AnalysisError, match="cannot order every use"):
        _analyzed(name)
