"""The typed analysis families, exercised through the composed operation.

One semantic invariant per family, and per defect: what a family reports for a
real model is the corpus Analyze witness's subject, and whether it runs at all is
every other test's. What is left here is the arithmetic and the failure
semantics -- the work model from both sides, the two capacity verdicts that read
differently, the bound that must not be summed, and the report that must not
disagree with itself.
"""

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
from tilefoundry.ir.types import DType, numel
from tilefoundry.target import AmxTarget, CudaTarget
from tilefoundry.target.facts import TARGET_FACTS


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


@module(entry="main", target="cuda")
class _Rotated:
    topologies = (Topology("cta", 1),)

    @func
    def main(
        q: Tensor[(1, 4, 2, 8), "f32"],
        k: Tensor[(1, 4, 2, 8), "f32"],
        cos_cache: ConstTensor[(4, 8), "f32"],
        sin_cache: ConstTensor[(4, 8), "f32"],
        pos_ids: ConstTensor[(4,), "i32"],
    ):
        rotated, _ = tf.rope(q, k, cos_cache, sin_cache, pos_ids)
        return rotated


@module(entry="main", target="cuda")
class _Allocated:
    topologies = (Topology("cta", 1),)

    @func
    def main(source: Tensor[(64,), "f32"]):
        return tf.add(source, tf.zeros(shape=(64,), dtype="f32"))


@module(entry="main", target="cuda")
class _BatchedOnTheRight:
    topologies = (Topology("cta", 1),)

    @func
    def main(
        token: Tensor[(1, 1, 4), "f32"],
        blocks: Tensor[(5, 4, 3), "f32"],
    ):
        return tf.matmul(token, blocks)


@func(target="cuda", topologies=(Topology("cta", 168),))
def _wide_grid(source: Tensor[(1024,), "f32"]):
    return tf.add(source, source)


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


def test_compute_cost_scales_leaf_work_by_the_whole_execution_mesh() -> None:
    """One authored call runs once per point of the topology hierarchy."""
    _, entry = _run(_thread_sharded, "compute-cost")

    record = get_metadata(_calls(entry)[-1], ComputeCostMetadata)
    assert record is not None
    assert record.execution_count == 8
    assert record.flops == (("f32", 16),)


def test_a_call_no_mesh_placed_runs_once_and_costs_the_same_on_either_machine() -> None:
    """An unsharded type states the whole tensor, so nothing multiplies it.

    The two readings of a tensor type are what makes this a real question. A
    sharded type states the extent one point holds, so recovering the whole means
    multiplying by the hierarchy -- which is what the test above measures. An
    unsharded type already states the whole, and folding the hierarchy into it reads
    the second as the first: an authored norm over `hidden` elements comes back
    multiplied by every thread the target declares, in units the traffic beside it
    is not counted in.

    Which is also why the same program is asked of two unrelated machines here:
    the work count is a property of the program. It used to be four times too
    large on either target -- the same four times, so an equality between the two
    still held while the number said the work of an unsharded add depended on how
    many blocks the target declares.
    """
    cuda_entry = _CudaAdd.entry_function()
    amx_entry = _AmxAdd.entry_function()

    analyze(_CudaAdd, cuda_entry, analysis="compute-cost")
    analyze(_AmxAdd, amx_entry, analysis="compute-cost")

    call = _calls(cuda_entry)[-1]
    record = get_metadata(call, ComputeCostMetadata)
    assert record is not None
    assert record.execution_count == 1
    assert record.flops == (("f32", numel(call.type)),)
    assert _cost_of(cuda_entry) == _cost_of(amx_entry)


def test_a_matmul_takes_its_batch_from_what_it_produced() -> None:
    """Either operand may be the broadcast one, so the output decides the batch.

    A block of a weight matrix multiplied by one token has its batch on the right:
    reading the left gave a batch of one and charged a whole block loop's arithmetic
    as a single tile's. The output's batch is what the call produced, and every batch
    of it was computed.
    """
    entry = _BatchedOnTheRight.entry_function()
    analyze(_BatchedOnTheRight, entry, analysis="compute-cost")

    record = get_metadata(_calls(entry)[-1], ComputeCostMetadata)
    assert record is not None
    # 5 batches of [1, 4] @ [4, 3]; the left operand states one batch.
    assert record.flops == (("f32", 2 * 5 * 1 * 4 * 3),)


def test_the_two_operations_a_decoder_stops_on_cost_what_they_do() -> None:
    """A rotation and an allocation, each derived from the operation itself.

    RoPE returns q and k together, so one call does two tensors' work: each
    element takes a multiply by its cosine, a multiply by its partner's sine, and
    the add between them. Allocating zeros is the other shape entirely -- it
    writes its whole output and computes nothing -- so it is asserted on the
    traffic side, where a cost model that priced an allocation as arithmetic, or
    as nothing at all, reads differently.
    """
    rotated = _Rotated.entry_function()
    analyze(_Rotated, rotated, analysis="compute-cost")
    rotation = get_metadata(
        next(call for call in _calls(rotated) if type(call.target).__name__ == "RoPE"),
        ComputeCostMetadata,
    )
    assert rotation is not None
    # 1 * 4 * 2 * 8 elements in each of q and k, at three flops apiece.
    assert rotation.flops == (("f32", 3 * 2 * 64),)

    allocated = _Allocated.entry_function()
    analyze(_Allocated, allocated, analysis="compute-cost")
    zeros = get_metadata(
        next(call for call in _calls(allocated) if type(call.target).__name__ == "Zeros"),
        ComputeCostMetadata,
    )
    assert zeros is not None
    assert zeros.flops == ()
    traffic = zeros.traffic_at("gmem")
    assert traffic.write_bytes == 64 * 4
    assert traffic.read_bytes == 0


def test_memory_holds_a_weight_resident_past_its_last_reader() -> None:
    """AC-1-2: a constant weight is never reclaimable within the function, while
    an ordinary value is live only between its definition and its last use."""
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
    assert any(item.last_used_at > item.defined_at for item in ordinary)


def test_an_addressable_level_that_overflows_fails_the_analysis() -> None:
    """AC-1-3: too much shared memory is an invalid program, not a warning."""
    with pytest.raises(AnalysisError, match="smem needs"):
        _run(_oversized_shared, "memory")


def test_a_cache_too_small_is_advisory_and_only_where_the_scopes_agree() -> None:
    """AC-1-3: an over-full cache costs speed, so the analysis still succeeds.

    And it is only reported where the comparison means something. A per-SM
    capacity set against a whole-device footprint exceeds it for almost any
    program, so reporting that would be noise rather than a finding: the per-SM
    share of that footprint is not known here, which is why l2 advises and l1
    stays silent about the same program.
    """
    result, entry = _run(_oversized_working_set, "memory")

    record = get_metadata(entry, MemoryMetadata)
    assert record is not None
    assert MemoryMetadata in result.metadata_types
    assert any("l2 holds" in note for note in record.advisories)
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


def test_roofline_reads_the_recorded_work_and_aggregates_before_dividing() -> None:
    """AC-1-4: the bound follows whatever compute-cost recorded, at the rate the
    target states.

    Aggregating the work first is what keeps the bound a lower bound. Each call's
    own bound rounds up to whole nanoseconds; adding those up would charge the
    function once per call for rounding the hardware never does, so the function's
    sides are summed in flops and bytes and divided once.
    """
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

    mixed = _MixedPrecision.entry_function()
    analyze(_MixedPrecision, mixed, analysis="roofline")
    whole = get_metadata(mixed, RooflineMetadata)
    assert whole is not None
    per_call = [
        record
        for call in _calls(mixed)
        if (record := get_metadata(call, RooflineMetadata)) is not None
    ]
    assert len(per_call) > 1
    assert whole.compute_ns < sum(item.compute_ns for item in per_call)
    assert whole.theoretical_ns == max(whole.compute_ns, whole.memory_ns)


def test_a_grid_wider_than_the_parallel_capacity_runs_in_waves() -> None:
    """One launch cannot occupy more units than the target admits at once, and
    the function-level record spans the whole plan."""
    _, entry = _run(_wide_grid, "timeline")

    record = get_metadata(_calls(entry)[-1], TimelineMetadata)
    assert record is not None
    assert record.grid_units == 168
    assert record.waves == 2

    whole = get_metadata(entry, TimelineMetadata)
    assert whole is not None
    assert whole.start_ns == 0
    ends = [
        record.end_ns
        for call in _calls(entry)
        if (record := get_metadata(call, TimelineMetadata)) is not None
    ]
    assert whole.end_ns == max(ends)


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


def test_the_gpu_memory_graph_is_not_a_tree() -> None:
    """AC-1-5: L1 caches L2, L2 caches GMEM, and L1 shares a block with SMEM.

    The shared edge is what makes the graph non-hierarchical: smem is an
    addressable level and l1 a cache, yet neither contains the other. A machine
    with no such block describes a plain cache chain in the same shape rather
    than carrying a stub edge, which is the case that would otherwise be modelled
    by a special one.
    """
    facts = TARGET_FACTS.project(CudaTarget(), MemoryHierarchyFacts)

    assert facts.cached_level("l1") == "l2"
    assert facts.cached_level("l2") == "gmem"
    assert facts.backing_level("l1") == "gmem"
    assert facts.capacity_sharers("l1") == (
        ("smem", CudaTarget().architecture.unified_l1_shared_per_sm_bytes),
    )
    assert {level.name for level in facts.explicit_levels} >= {"gmem", "smem", "rmem"}
    assert {level.name for level in facts.implicit_levels} == {"l1", "l2"}

    flat = TARGET_FACTS.project(AmxTarget(), MemoryHierarchyFacts)
    assert flat.cached_level("l1d") == "l2"
    assert flat.backing_level("l1d") == "gmem"
    assert flat.capacity_sharers("l1d") == ()
    assert all(
        relation.kind is MemoryRelationKind.CACHES for relation in flat.relations
    )


def test_a_report_shows_the_requested_analyses_and_reads_the_same_either_way() -> None:
    """A requested root pulls its dependencies in, so their records land on the
    IR without having been asked for. A report shows what was requested; the
    dependency's records stay on the IR for whoever does ask.

    Two formats over one structure cannot disagree about a conclusion, so the
    text is checked to carry the verdict the JSON states.
    """
    entry = _entry(_CudaAdd)
    result = analyze(_CudaAdd, entry, analysis="roofline")

    data = report([result])

    assert data["executed"] == ["compute-cost", "memory", "roofline"]
    assert set(data["function_records"]) == {"roofline"}
    assert get_metadata(entry, MemoryMetadata) is not None

    payload = json.loads(render_json(data))
    assert payload == data
    assert f"by={payload['function_records']['roofline']['bound_by']}" in render_text(data)
