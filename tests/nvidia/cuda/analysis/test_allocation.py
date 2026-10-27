"""Physical placement, semantic aliases, and call-boundary proofs."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.analysis import (
    MemoryMetadata,
    RegionMemoryMetadata,
    analyze,
)
from tilefoundry.analysis.access import AliasKind
from tilefoundry.analysis.allocation import AllocationSummaryMetadata
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.liveness import LivenessMetadata
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.schedule import finalize
from tilefoundry.visitor_registry.verify import verify_module

ANALYSIS = Path(__file__).parents[1] / "fixtures" / "analysis"


class AllocationCase(Case):
    """A program whose memory analysis places, aliases, or keeps its values resident."""


@case
class SequentialCarries(AllocationCase):
    """An insert slice keeps its address through sequential carry loops."""

    path: Path = Path(__file__).parents[1] / "fixtures/logical/sequential_carries.py"

    def guard(self, state, problems):
        result = analyze(*check_program(state, state.entry_function()), analysis="memory")
        inserts = [
            value
            for value in collect_exprs(result.function.body)
            if isinstance(value, Call) and isinstance(value.target, InsertSlice)
        ]
        assert len(inserts) == 3
        values = {
            value.value: value
            for value in get_metadata(result.function, AllocationSummaryMetadata).values
        }
        for insert in inserts:
            memory = get_metadata(insert, MemoryMetadata)
            link = next(
                alias
                for alias in values[SSAValue(insert)].aliases
                if alias.kind is AliasKind.INSERT_SLICE
            )
            assert link.relation.relation.is_identity()
            assert link.target.expr is insert.args[0]
            assert memory.buffer_bytes is None and memory.offsets == ()
        verify_module(finalize(result))


@dataclass(frozen=True)
class FreshReturn(AllocationCase):
    """A retained elementwise return overwrites only a dead actual."""

    later_use: bool = False

    def guard(self, state, problems):
        result = analyze(*check_program(state, state.entry_function()), analysis="memory")
        call = next(
            value
            for value in collect_exprs(result.function.body)
            if isinstance(value, Call) and isinstance(value.target, Function)
        )
        actual = get_metadata(call.args[0], MemoryMetadata)
        returned = get_metadata(call, MemoryMetadata)
        assert actual.offsets and returned.offsets
        assert (returned.offsets == actual.offsets) is (not self.later_use)
        assert returned.buffer_bytes == actual.buffer_bytes == 32
        summary = get_metadata(result.function, AllocationSummaryMetadata)
        values = {value.value: value for value in summary.values}
        assert (
            AllocationSummaryMetadata(tuple(values.values())).root(values[SSAValue(call)])
            is AllocationSummaryMetadata(tuple(values.values())).root(
                values[SSAValue(call.args[0])]
            )
        ) is (not self.later_use)


@case
class FreshReturnDeadActual(FreshReturn):
    path: Path = Path(__file__).parents[1] / "fixtures/logical/fresh_return_0.py"


@case
class FreshReturnLiveActual(FreshReturn):
    path: Path = Path(__file__).parents[1] / "fixtures/logical/fresh_return_1.py"
    later_use: bool = True


@case
class DynamicSelection(AllocationCase):
    """A captured dynamic view pins both possible leaves without enabling reuse."""

    path: Path = ANALYSIS / "dynamic_selection.py"
    name: str | None = "DynamicSelection"

    def guard(self, state, problems):
        result = analyze(
            *check_program(state, state.entry_function()),
            analysis="memory",
        )
        selected = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, TupleGetItem)
        )
        first, second = selected.args[0].elements
        consumer = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call)
            and len(expr.args) == 2
            and all(isinstance(arg, MeshRegion) for arg in expr.args)
        )
        records = [get_metadata(value, MemoryMetadata) for value in (first, second, consumer)]
        assert all(record.buffer_bytes == 8 and len(record.offsets) == 1 for record in records)
        spans = sorted(
            (record.offsets[0], record.offsets[0] + record.buffer_bytes) for record in records
        )
        assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
        moved = get_metadata(result.function, RegionMemoryMetadata)
        assert moved.peak_for("smem").peak_bytes == 40
        lifetimes = {life.value: life for life in moved.lifetimes}
        output = lifetimes[SSAValue(consumer)]
        assert all(
            lifetimes[SSAValue(leaf)].last_used_at < output.defined_at for leaf in (first, second)
        )
        assert get_metadata(selected, MemoryMetadata).buffer_bytes is None
        assert not get_metadata(selected, MemoryMetadata).offsets
        summary = get_metadata(result.function, AllocationSummaryMetadata)
        by_value = {value.value: value for value in summary.values}
        assert not by_value[SSAValue(consumer)].aliases
        assert SSAValue(selected) not in by_value
        liveness = get_metadata(result.function, LivenessMetadata)
        for index in range(len(selected.args[0].elements)):
            assert (
                liveness.interval_of(SSAValue(selected.args[0], index)).last_used_at
                == liveness.timeline_end
            )
        for region in consumer.args:
            assert by_value.get(SSAValue(region)) is None
            assert all(by_value.get(SSAValue(param)) is None for param in region.params)
        allocated = by_value[SSAValue(consumer)]
        assert all(
            summary.root(allocated) is not summary.root(SSAValue(leaf)) for leaf in (first, second)
        )


CASES = special(AllocationCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_allocation_case(case: AllocationCase) -> None:
    check_case(case)
