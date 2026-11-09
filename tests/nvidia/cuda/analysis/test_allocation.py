"""Physical placement, semantic aliases, and call-boundary proofs."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry import analysis, verifier
from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import tensor
from tilefoundry.schedule import finalize

ANALYSIS = Path(__file__).parents[1] / "fixtures" / "analysis"


class AllocationCase(Case):
    """A program whose memory analysis places, aliases, or keeps its values resident."""


@case
class SequentialCarries(AllocationCase):
    """An insert slice keeps its address through sequential carry loops."""

    path: Path = Path(__file__).parents[1] / "fixtures/logical/sequential_carries.py"

    def guard(self, state, problems):
        result = analysis.analyze(*analysis.check_program(state, state.entry_function()), analysis=("memory", "sync"))
        inserts = [
            value
            for value in core.collect_exprs(result.function.body)
            if isinstance(value, expr.Call) and isinstance(value.target, tensor.InsertSlice)
        ]
        assert len(inserts) == 3
        values = {
            value.value: value
            for value in expr.get_metadata(result.function, analysis.AllocationSummaryMetadata).values
        }
        for insert in inserts:
            memory = expr.get_metadata(insert, analysis.MemoryMetadata)
            link = next(
                alias
                for alias in values[core.SSAValue(insert)].aliases
                if alias.kind is analysis.AliasKind.INSERT_SLICE
            )
            assert link.relation.relation.is_identity()
            assert link.target.expr is insert.args[0]
            assert memory.buffer_bytes is None and memory.offsets == ()
        verifier.verify_module(finalize(result))


@dataclass(frozen=True)
class FreshReturn(AllocationCase):
    """A retained elementwise return overwrites only a dead actual."""

    later_use: bool = False

    def guard(self, state, problems):
        result = analysis.analyze(*analysis.check_program(state, state.entry_function()), analysis="memory")
        call = next(
            value
            for value in core.collect_exprs(result.function.body)
            if isinstance(value, expr.Call) and isinstance(value.target, core.Function)
        )
        actual = expr.get_metadata(call.args[0], analysis.MemoryMetadata)
        returned = expr.get_metadata(call, analysis.MemoryMetadata)
        assert actual.offsets and returned.offsets
        assert (returned.offsets == actual.offsets) is (not self.later_use)
        assert returned.buffer_bytes == actual.buffer_bytes == 32
        summary = expr.get_metadata(result.function, analysis.AllocationSummaryMetadata)
        values = {value.value: value for value in summary.values}
        assert (
            analysis.AllocationSummaryMetadata(tuple(values.values())).root(values[core.SSAValue(call)])
            is analysis.AllocationSummaryMetadata(tuple(values.values())).root(
                values[core.SSAValue(call.args[0])]
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
        result = analysis.analyze(
            *analysis.check_program(state, state.entry_function()),
            analysis="memory",
        )
        selected = next(
            node
            for node in core.collect_exprs(result.function.body)
            if isinstance(node, expr.Call) and isinstance(node.target, tensor.TupleGetItem)
        )
        first, second = selected.args[0].elements
        consumer = next(
            node
            for node in core.collect_exprs(result.function.body)
            if isinstance(node, expr.Call)
            and len(node.args) == 2
            and all(isinstance(arg, core.MeshRegion) for arg in node.args)
        )
        records = [expr.get_metadata(value, analysis.MemoryMetadata) for value in (first, second, consumer)]
        assert all(record.buffer_bytes == 8 and len(record.offsets) == 1 for record in records)
        spans = sorted(
            (record.offsets[0], record.offsets[0] + record.buffer_bytes) for record in records
        )
        assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
        moved = expr.get_metadata(result.function, analysis.RegionMemoryMetadata)
        assert moved.peak_for("smem").peak_bytes == 40
        lifetimes = {life.value: life for life in moved.lifetimes}
        output = lifetimes[core.SSAValue(consumer)]
        assert all(
            lifetimes[core.SSAValue(leaf)].last_used_at < output.defined_at for leaf in (first, second)
        )
        assert expr.get_metadata(selected, analysis.MemoryMetadata).buffer_bytes is None
        assert not expr.get_metadata(selected, analysis.MemoryMetadata).offsets
        summary = expr.get_metadata(result.function, analysis.AllocationSummaryMetadata)
        by_value = {value.value: value for value in summary.values}
        assert not by_value[core.SSAValue(consumer)].aliases
        assert core.SSAValue(selected) not in by_value
        liveness = expr.get_metadata(result.function, analysis.LivenessMetadata)
        for index in range(len(selected.args[0].elements)):
            assert (
                liveness.interval_of(core.SSAValue(selected.args[0], index)).last_used_at
                == liveness.timeline_end
            )
        for region in consumer.args:
            assert by_value.get(core.SSAValue(region)) is None
            assert all(by_value.get(core.SSAValue(param)) is None for param in region.params)
        allocated = by_value[core.SSAValue(consumer)]
        assert all(
            summary.root(allocated) is not summary.root(core.SSAValue(leaf)) for leaf in (first, second)
        )


CASES = special(AllocationCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_allocation_case(case: AllocationCase) -> None:
    check_case(case)
