"""The typed analysis families, exercised through the composed operation."""

from __future__ import annotations

import json

import pytest

from tilefoundry import func, module
from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryHierarchyFacts,
    MemoryMetadata,
    MemoryRelationKind,
    RooflineMetadata,
    ThroughputFacts,
    TimelineMetadata,
)
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.walk import postorder
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, tf
from tilefoundry.inspection.analysis_report import render_json, render_text, report
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Layout
from tilefoundry.target import AmxTarget, CudaTarget
from tilefoundry.target.facts import TARGET_FACTS

_THREAD_MESH = Mesh(Topology("thread", 4), Layout((4,), (1,)), names=("lane",))


@module(entry="main", target="cuda")
class _CudaAdd:
    topologies = (Topology("cta", 4),)

    @func
    def main(source: Tensor[(256,), "f32"]):
        return tf.add(source, source)


@module(entry="main", target="amx")
class _AmxAdd:
    topologies = (Topology("core", 4),)

    @func
    def main(source: Tensor[(256,), "f32"]):
        return tf.add(source, source)


@module(entry="main", target="cuda")
class _MixedPrecision:
    topologies = (Topology("cta", 1),)

    @func
    def main(source: Tensor[(64,), "f32"]):
        scaled = tf.mul(source, source)
        half = tf.cast(scaled, "f16")
        return tf.add(half, half)


@module(entry="main", target="cuda")
class _WeightedAdd:
    topologies = (Topology("cta", 1),)

    @func
    def main(
        source: Tensor[(64,), "f32"],
        weight: ConstTensor[(64,), "f32"],
    ):
        scaled = tf.mul(source, weight)
        return tf.add(scaled, scaled)


@func(target="cuda", topologies=(Topology("cta", 168),))
def _wide_grid(source: Tensor[(1024,), "f32"]):
    return tf.add(source, source)


@func(target="cuda", topologies=(Topology("cta", 64),))
def _independent_branches(source: Tensor[(1,), "f32"]):
    return tf.add(source, source), tf.mul(source, source)


@func(target="cuda", topologies=(Topology("thread", 1),))
def _reshard_boundary(source: Tensor[(1,), "f32"]):
    with Mesh(topology="thread", layout=(1,), names=("lane",)) as thread:
        local = tf.reshard(source, (1 @ thread.lane,), "rmem")
        moved = tf.reshard(local, (1 @ thread.lane,), "rmem")
        return tf.add(moved, moved)


@func(target="cuda", topologies=(Topology("cta", 2), Topology("thread", 4)))
def _thread_sharded(source: Tensor[(8,), "f32"]):
    with Mesh(topology="thread", layout=(4,), names=("lane",)) as thread:
        local = tf.reshard(source, (8 @ thread.lane,), "rmem")
        return tf.add(local, local)


@module(entry="caller", target="cuda")
class _NestedCall:
    topologies = (Topology("thread", 4),)

    @func
    def helper(source: Tensor[(8,), "f32", (8 @ _THREAD_MESH.lane,), "rmem"]):
        return tf.add(source, source)

    @func
    def caller(source: Tensor[(8,), "f32", (8 @ _THREAD_MESH.lane,), "rmem"]):
        return helper(source)


@func(target="cuda", topologies=(Topology("thread", 4),))
def _oversized_shared(source: Tensor[(131072,), "f32"]):
    with Mesh(topology="thread", layout=(4,), names=("lane",)) as thread:
        local = tf.reshard(source, (131072 @ thread.lane,), "smem")
        return tf.add(local, local)


@func(target="cuda", topologies=(Topology("thread", 4),))
def _modest_shared(source: Tensor[(1024,), "f32"]):
    with Mesh(topology="thread", layout=(4,), names=("lane",)) as thread:
        local = tf.reshard(source, (1024 @ thread.lane,), "smem")
        return tf.add(local, local)


@func(target="cuda", topologies=(Topology("cta", 1),))
def _oversized_working_set(source: Tensor[(16_000_000,), "f32"]):
    return tf.add(source, source)


def _entry(owner):
    """The Function a Module is entered through."""
    return owner.entry_function()


def _run(owner, analysis: str):
    """Analyse *owner*'s entry function and hand back that function."""
    entry = _entry(owner)
    return analyze(owner, entry, analysis=analysis), entry


def _calls(function) -> tuple[Call, ...]:
    return tuple(expr for expr in postorder(function.body) if isinstance(expr, Call))


def _cost_of(function) -> ComputeCostMetadata:
    (record,) = [
        get_metadata(call, ComputeCostMetadata) for call in _calls(function)[-1:]
    ]
    assert record is not None
    return record


def test_compute_cost_reports_the_same_work_on_unrelated_targets() -> None:
    """AC-1-1: the work count is a property of the program, not the machine."""
    cuda_entry = _CudaAdd.entry_function()
    amx_entry = _AmxAdd.entry_function()

    analyze(_CudaAdd, cuda_entry, analysis="compute-cost")
    analyze(_AmxAdd, amx_entry, analysis="compute-cost")

    assert _cost_of(cuda_entry) == _cost_of(amx_entry)
    assert _cost_of(cuda_entry).flops == (("f32", 1024),)


def test_compute_cost_keeps_each_dtype_separate() -> None:
    """AC-1-1: mixed precision has no single flop count to collapse into."""
    entry = _MixedPrecision.entry_function()

    analyze(_MixedPrecision, entry, analysis="compute-cost")

    counted = {
        name
        for call in _calls(entry)
        if (record := get_metadata(call, ComputeCostMetadata)) is not None
        for name, _value in record.flops
    }
    assert counted == {"f16", "f32"}


def test_compute_cost_scales_leaf_work_by_the_whole_execution_mesh() -> None:
    """One authored call runs once per point of the topology hierarchy."""
    _, entry = _run(_thread_sharded, "compute-cost")

    record = get_metadata(_calls(entry)[-1], ComputeCostMetadata)
    assert record is not None
    assert record.execution_count == 8
    assert record.flops == (("f32", 16),)


def test_a_call_into_a_function_costs_what_that_function_costs() -> None:
    """A nested call reports the callee's totals rather than nothing."""
    entry = _NestedCall.entry_function()

    analyze(_NestedCall, entry, analysis="compute-cost")

    call = entry.body
    assert isinstance(call, Call)
    record = get_metadata(call, ComputeCostMetadata)
    assert record is not None
    # Two f32 elements per thread, over the four threads the mesh declares.
    assert record.flops == (("f32", 8),)


def test_memory_holds_a_weight_resident_past_its_last_reader() -> None:
    """AC-1-2: a constant weight is never reclaimable within the function."""
    entry = _WeightedAdd.entry_function()
    analyze(_WeightedAdd, entry, analysis="memory")

    record = get_metadata(entry, MemoryMetadata)
    assert record is not None
    weights = [item for item in record.lifetimes if item.persistent]
    ordinary = [item for item in record.lifetimes if not item.persistent]
    assert [item.binding for item in weights] == ["weight"]
    assert ordinary, "the other values are still measured by their live ranges"

    # The weight is read early and still held to the end; an ordinary value that
    # is read as early releases at that point.
    (held,) = weights
    assert held.last_used_at == max(item.last_used_at for item in record.lifetimes)
    assert any(item.last_used_at < held.last_used_at for item in ordinary)


def test_memory_measures_temporaries_by_first_definition_and_last_use() -> None:
    """AC-1-2: an ordinary value is live only between its definition and use."""
    entry = _MixedPrecision.entry_function()
    analyze(_MixedPrecision, entry, analysis="memory")

    record = get_metadata(entry, MemoryMetadata)
    assert record is not None
    assert record.lifetimes
    assert all(not item.persistent for item in record.lifetimes)
    assert any(item.last_used_at > item.defined_at for item in record.lifetimes)


def test_an_addressable_level_that_overflows_fails_the_analysis() -> None:
    """AC-1-3: too much shared memory is an invalid program, not a warning."""
    with pytest.raises(AnalysisError, match="smem needs"):
        _run(_oversized_shared, "memory")


def test_a_cache_that_cannot_hold_the_working_set_is_only_advisory() -> None:
    """AC-1-3: an over-full cache costs speed, so the analysis still succeeds."""
    result, entry = _run(_oversized_working_set, "memory")

    record = get_metadata(entry, MemoryMetadata)
    assert record is not None
    assert MemoryMetadata in result.metadata_types
    assert any("l2 holds" in note for note in record.advisories)


def test_a_cache_is_not_compared_against_a_footprint_of_another_scope() -> None:
    """A per-SM capacity set against a whole-device footprint exceeds it for
    almost any program, so reporting that would be noise rather than a finding:
    the per-SM share of that footprint is not known here."""
    _, entry = _run(_oversized_working_set, "memory")

    record = get_metadata(entry, MemoryMetadata)
    assert record is not None
    assert not any("l1 holds" in note for note in record.advisories)


def test_a_shared_block_reports_what_the_program_leaves_the_cache() -> None:
    """AC-1-5: the sharing edge is what makes L1's usable size depend on the
    program, and that division is reportable without any working set."""
    _, entry = _run(_modest_shared, "memory")

    record = get_metadata(entry, MemoryMetadata)
    assert record is not None
    block = CudaTarget().architecture.unified_l1_shared_per_sm_bytes
    smem = record.level("smem")
    assert smem is not None and smem.peak_bytes > 0
    assert any(
        f"smem claims {smem.peak_bytes} B of the {block} B block" in note
        and f"leaving l1 {block - smem.peak_bytes} B" in note
        for note in record.advisories
    )


def test_roofline_reads_the_recorded_work_rather_than_recounting_it() -> None:
    """AC-1-4: the bound follows whatever compute-cost recorded."""
    entry = _CudaAdd.entry_function()
    result = analyze(_CudaAdd, entry, analysis="roofline")

    assert result.executed == ("compute-cost", "memory", "roofline")
    call = _calls(entry)[-1]
    cost = get_metadata(call, ComputeCostMetadata)
    bound = get_metadata(call, RooflineMetadata)
    assert cost is not None and bound is not None

    facts = TARGET_FACTS.project(CudaTarget(), ThroughputFacts)
    rate = facts.peak_for(DType.f32)
    assert rate is not None
    expected = -(-cost.flops[0][1] * 1_000_000_000 // rate)
    assert bound.compute_ns == expected


def test_a_function_bound_is_not_the_sum_of_the_call_bounds() -> None:
    """Aggregating the work first is what keeps the bound a lower bound.

    Each call's own bound rounds up to whole nanoseconds. Adding those up would
    charge the function once per call for rounding the hardware never does, so
    the function's sides are summed in flops and bytes and divided once.
    """
    entry = _MixedPrecision.entry_function()
    analyze(_MixedPrecision, entry, analysis="roofline")

    whole = get_metadata(entry, RooflineMetadata)
    assert whole is not None
    per_call = [
        record
        for call in _calls(entry)
        if (record := get_metadata(call, RooflineMetadata)) is not None
    ]
    assert len(per_call) > 1
    assert whole.compute_ns < sum(item.compute_ns for item in per_call)
    assert whole.theoretical_ns == max(whole.compute_ns, whole.memory_ns)


def test_a_grid_wider_than_the_parallel_capacity_runs_in_waves() -> None:
    """One launch cannot occupy more units than the target admits at once."""
    _, entry = _run(_wide_grid, "timeline")

    record = get_metadata(_calls(entry)[-1], TimelineMetadata)
    assert record is not None
    assert record.grid_units == 168
    assert record.waves == 2


def test_independent_units_are_placed_at_the_same_time() -> None:
    """Two units with no dependency between them need not be ordered."""
    _, entry = _run(_independent_branches, "timeline")

    routed, shared = _calls(entry)[:2]
    first = get_metadata(routed, TimelineMetadata)
    second = get_metadata(shared, TimelineMetadata)
    assert first is not None and second is not None
    assert first.start_ns == second.start_ns == 0


def test_a_reshard_ends_the_execution_unit_on_both_sides() -> None:
    """A reshard exists to move data, so no unit may span one."""
    _, entry = _run(_reshard_boundary, "timeline")

    local, moved, consumer = _calls(entry)[:3]
    assert isinstance(local.target, Reshard)
    assert isinstance(moved.target, Reshard)
    records = [get_metadata(call, TimelineMetadata) for call in (local, moved, consumer)]
    assert all(record is not None for record in records)
    assert records[0].end_ns <= records[1].start_ns
    assert records[1].end_ns <= records[2].start_ns


def test_the_function_timeline_reports_the_solved_makespan() -> None:
    """The function-level record spans the whole plan."""
    _, entry = _run(_wide_grid, "timeline")

    whole = get_metadata(entry, TimelineMetadata)
    assert whole is not None
    assert whole.start_ns == 0
    ends = [
        record.end_ns
        for call in _calls(entry)
        if (record := get_metadata(call, TimelineMetadata)) is not None
    ]
    assert whole.end_ns == max(ends)


def test_the_gpu_memory_graph_is_not_a_tree() -> None:
    """AC-1-5: L1 caches L2, L2 caches GMEM, and L1 shares a block with SMEM."""
    facts = TARGET_FACTS.project(CudaTarget(), MemoryHierarchyFacts)

    assert facts.cached_level("l1") == "l2"
    assert facts.cached_level("l2") == "gmem"
    assert facts.backing_level("l1") == "gmem"
    assert facts.capacity_sharers("l1") == (
        ("smem", CudaTarget().architecture.unified_l1_shared_per_sm_bytes),
    )
    # The shared edge is what makes the graph non-hierarchical: smem is an
    # addressable level and l1 a cache, yet neither contains the other.
    assert {level.name for level in facts.explicit_levels} >= {"gmem", "smem", "rmem"}
    assert {level.name for level in facts.implicit_levels} == {"l1", "l2"}


def test_a_target_without_shared_capacity_says_so_in_the_same_shape() -> None:
    """AC-1-5: the flat graph describes a plain cache chain without a stub edge."""
    facts = TARGET_FACTS.project(AmxTarget(), MemoryHierarchyFacts)

    assert facts.cached_level("l1d") == "l2"
    assert facts.backing_level("l1d") == "gmem"
    assert facts.capacity_sharers("l1d") == ()
    assert all(
        relation.kind is MemoryRelationKind.CACHES for relation in facts.relations
    )


def test_a_report_shows_the_requested_analyses_not_every_written_record() -> None:
    """A requested root pulls its dependencies in, so their records land on the
    IR without having been asked for. A report shows what was requested; the
    dependency's records stay on the IR for whoever does ask."""
    entry = _entry(_CudaAdd)
    result = analyze(_CudaAdd, entry, analysis="roofline")

    data = report([result])

    assert data["executed"] == ["compute-cost", "memory", "roofline"]
    assert set(data["function_records"]) == {"roofline"}
    assert get_metadata(entry, MemoryMetadata) is not None


def test_a_report_covers_every_analysis_it_was_given() -> None:
    """Asking for two roots reports both, and each root's own records."""
    entry = _entry(_CudaAdd)
    results = [
        analyze(_CudaAdd, entry, analysis=name) for name in ("memory", "timeline")
    ]

    data = report(results)

    assert data["requested"] == ["memory", "timeline"]
    assert set(data["function_records"]) == {"memory", "timeline"}


def test_json_and_text_render_the_same_report() -> None:
    """Two formats over one structure cannot disagree about a conclusion."""
    entry = _entry(_CudaAdd)
    data = report([analyze(_CudaAdd, entry, analysis="roofline")])

    payload = json.loads(render_json(data))
    text = render_text(data)

    assert payload == data
    assert f"by={payload['function_records']['roofline']['bound_by']}" in text
