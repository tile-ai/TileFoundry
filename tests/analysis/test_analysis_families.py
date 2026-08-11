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
from tilefoundry.ir.core import Call, Var, get_metadata
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import DType, make_tensor_type, numel
from tilefoundry.target import AmxTarget, CudaTarget
from tilefoundry.visitor_registry import cost_evaluator_registry
from tilefoundry.visitor_registry.contexts import Cost, CostContext, TrafficBytes
from tilefoundry.visitor_registry.visitors import CostEvaluator


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),))
class _CudaAdd:
    @func
    def main(source: Tensor[(256,), "f32"]):
        return tf.add(source, source)


@module(entry="main", target=AmxTarget(), topologies=(Topology("core", 4),))
class _AmxAdd:
    @func
    def main(source: Tensor[(256,), "f32"]):
        return tf.add(source, source)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _MixedPrecision:
    @func
    def main(source: Tensor[(64,), "f32"]):
        scaled = tf.mul(source, source)
        half = tf.cast(scaled, "f16")
        return tf.add(half, half)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _WeightedAdd:
    @func
    def main(
        source: Tensor[(64,), "f32"],
        weight: ConstTensor[(64,), "f32"],
    ):
        scaled = tf.mul(source, weight)
        return tf.add(scaled, scaled)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _Rotated:
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


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _Allocated:
    @func
    def main(source: Tensor[(64,), "f32"]):
        return tf.add(source, tf.zeros(shape=(64,), dtype="f32"))


@module(entry="row", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _MovementCosts:
    @func
    def row(source: Tensor[(1024, 2048), "f32"]):
        return source[0:256, :]

    @func
    def column(source: Tensor[(1024, 2048), "f32"]):
        return source[:, 0:512]

    @func
    def materialized(source: Tensor[(1024, 2048), "f32"]):
        transposed = tf.transpose(source, perm=(1, 0))
        return tf.reshape(transposed, new_shape=(1024 * 2048,))

    @func
    def copied(source: Tensor[(1024, 2048), "f32"]):
        selected = source[0:256, :]
        return tf.concat(selected, selected, axis=0)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _IndexSelected:
    @func
    def main(
        table: ConstTensor[(1024, 64), "f32"],
        rows: ConstTensor[(4,), "i32"],
    ):
        return tf.index_select(table, rows, dim=0)


@module(entry="plain", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 132),))
class _MatmulLayouts:
    @func
    def plain(
        lhs: Tensor[(1056, 2048), "bf16"],
        rhs: ConstTensor[(2048, 6600), "bf16"],
    ):
        return tf.matmul(lhs, rhs)

    @func
    def split(
        lhs: Tensor[(1056, 2048), "bf16"],
        rhs: ConstTensor[(2048, 6600), "bf16"],
    ):
        with Mesh(("cta",), layout=(132,), names=("tile",)) as cta:
            local_lhs = tf.reshard(lhs, (1056 @ cta.tile, 2048), "gmem")
            local_rhs = tf.reshard(rhs, (2048, 6600), "gmem")
            return tf.matmul(local_lhs, local_rhs)

    @func
    def broadcast(
        lhs: Tensor[(1056, 2048), "bf16"],
        rhs: ConstTensor[(2048, 6600), "bf16"],
    ):
        with Mesh(("cta",), layout=(132,), names=("tile",)) as _cta:
            local_lhs = tf.reshard(lhs, (1056, 2048), "gmem")
            local_rhs = tf.reshard(rhs, (2048, 6600), "gmem")
            return tf.matmul(local_lhs, local_rhs)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),))
class _NestedSplitAdd:
    @func
    def block(source: Tensor[(256,), "f32"]):
        with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
            local = tf.reshard(source, (256 @ cta.tile,), "gmem")
            return tf.add(local, local)

    @func
    def main(source: Tensor[(256,), "f32"]):
        return block(source)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 168),))
def _wide_grid(source: Tensor[(1024,), "f32"]):
    return tf.add(source, source)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("thread", 1),))
def _reshard_boundary(source: Tensor[(1,), "f32"]):
    with Mesh(("thread",), layout=(1,), names=("lane",)) as thread:
        local = tf.reshard(source, (1 @ thread.lane,), "rmem")
        moved = tf.reshard(local, (1 @ thread.lane,), "rmem")
        return tf.add(moved, moved)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 2), Topology("thread", 4)))
def _thread_sharded(source: Tensor[(8,), "f32"]):
    with Mesh(("thread",), layout=(4,), names=("lane",)) as thread:
        local = tf.reshard(source, (8 @ thread.lane,), "rmem")
        return tf.add(local, local)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", None),))
def _launch_provided_tiles(source: Tensor[(8, 128), "f32"]):
    with Mesh(("cta",), layout=(8,), names=("tile",)) as cta:
        local = tf.reshard(source, (8 @ cta.tile, 128), "rmem")
        return tf.add(local, local)


@func(
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 2)),
)
def _multi_topology_mesh(source: Tensor[(4,), "f32"]):
    with Mesh(("cta", "thread"), layout=(4,), names=("position",)) as mesh:
        local = tf.reshard(source, (4 @ mesh.position,), "rmem")
        return tf.add(local, local)


@module(entry="split", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 132),))
class _SharedTile:
    @func
    def split(source: Tensor[(1056, 6600), "f32"]):
        with Mesh(("cta",), layout=(132,), names=("tile",)) as cta:
            local = tf.reshard(source, (1056 @ cta.tile, 6600), "smem")
            return tf.add(local, local)

    @func
    def broadcast(source: Tensor[(1056, 6600), "f32"]):
        with Mesh(("cta",), layout=(132,), names=("tile",)) as _cta:
            local = tf.reshard(source, (1056, 6600), "smem")
            return tf.add(local, local)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("thread", 4),))
def _modest_shared(source: Tensor[(1024,), "f32"]):
    with Mesh(("thread",), layout=(4,), names=("lane",)) as thread:
        local = tf.reshard(source, (1024 @ thread.lane,), "smem")
        return tf.add(local, local)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
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
    (record,) = [get_metadata(call, ComputeCostMetadata) for call in _calls(function)[-1:]]
    assert record is not None
    return record


def test_compute_cost_stops_at_the_selected_topology_level() -> None:
    entry = _entry(_thread_sharded)
    cta_result = analyze(_thread_sharded, entry, analysis="compute-cost")

    record = get_metadata(_calls(entry)[-1], ComputeCostMetadata)
    assert record is not None
    assert cta_result.level == "cta"
    assert record.flops == (("f32", 8),)
    assert record.flops_per_unit == (("f32", 8),)
    traffic = record.traffic

    thread_result = analyze(_thread_sharded, entry, analysis="compute-cost", level="thread")
    record = get_metadata(_calls(entry)[-1], ComputeCostMetadata)
    assert record is not None
    assert thread_result.level == "thread"
    assert record.flops == (("f32", 8),)
    assert record.flops_per_unit == (("f32", 2),)
    assert record.traffic == traffic


def test_an_unsharded_call_reports_the_same_global_and_per_unit_work() -> None:
    cuda_entry = _CudaAdd.entry_function()
    amx_entry = _AmxAdd.entry_function()

    analyze(_CudaAdd, cuda_entry, analysis="compute-cost")
    analyze(_AmxAdd, amx_entry, analysis="compute-cost")

    call = _calls(cuda_entry)[-1]
    record = get_metadata(call, ComputeCostMetadata)
    assert record is not None
    assert record.flops == (("f32", numel(call.type)),)
    assert record.flops_per_unit == record.flops
    assert _cost_of(cuda_entry) == _cost_of(amx_entry)


def test_matmul_layout_changes_only_the_per_unit_work() -> None:
    records = []
    functions = {function.name: function for function in _MatmulLayouts.functions}
    for function in (functions["plain"], functions["split"], functions["broadcast"]):
        analyze(_MatmulLayouts, function, analysis="roofline")
        call = _calls(function)[-1]
        cost = get_metadata(call, ComputeCostMetadata)
        bound = get_metadata(call, RooflineMetadata)
        assert cost is not None and bound is not None
        records.append((cost, bound))

    (
        (plain_cost, plain_bound),
        (split_cost, split_bound),
        (
            broadcast_cost,
            broadcast_bound,
        ),
    ) = records
    assert [record.flops for record, _bound in records] == [
        (("bf16", 28_547_481_600),),
    ] * 3
    assert [record.flops_per_unit for record, _bound in records] == [
        (("bf16", 28_547_481_600),),
        (("bf16", 216_268_800),),
        (("bf16", 28_547_481_600),),
    ]
    assert split_cost.traffic == plain_cost.traffic == broadcast_cost.traffic
    assert split_bound == plain_bound == broadcast_bound
    assert plain_bound.ideal_ns == 28_851


def test_function_call_carries_the_callee_per_unit_work() -> None:
    entry = _NestedSplitAdd.entry_function()
    analyze(_NestedSplitAdd, entry, analysis="compute-cost")

    record = get_metadata(_calls(entry)[-1], ComputeCostMetadata)
    assert record is not None
    assert record.flops == (("f32", 256),)
    assert record.flops_per_unit == (("f32", 64),)


def test_a_launch_provided_topology_uses_its_mesh_layout_for_analysis() -> None:
    """A dynamic launch declares its positions in the mesh layout, not Topology."""
    _, entry = _run(_launch_provided_tiles, "timeline")

    call = _calls(entry)[-1]
    cost = get_metadata(call, ComputeCostMetadata)
    placement = get_metadata(call, TimelineMetadata)
    assert cost is not None and placement is not None
    assert cost.flops == (("f32", 8 * 128),)
    assert cost.flops_per_unit == (("f32", 128),)
    assert placement.grid_units == 8
    assert placement.waves == 1


def test_analysis_refuses_a_position_count_for_a_multi_topology_mesh() -> None:
    with pytest.raises(
        AnalysisError,
        match="one mesh names multiple topology levels",
    ):
        _run(_multi_topology_mesh, "compute-cost")


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
    assert traffic.write == 64 * 4
    assert traffic.read == 0


def test_index_select_reads_the_rows_it_names_and_not_the_table() -> None:
    result, entry = _run(_IndexSelected, "compute-cost")

    record = get_metadata(_calls(entry)[-1], ComputeCostMetadata)
    assert record is not None
    traffic = record.traffic_at("gmem")
    rows = 4 * 64 * 4
    assert traffic.read == rows + 4 * 4
    assert traffic.write == rows

    payload = json.loads(render_json(report([result])))
    table, indices, produced = payload["calls"][-1]["compute-cost"]["operands"]
    assert table == {
        "arg": 0,
        "name": "table",
        "type": "f32[1024,64] gmem",
        "read": rows,
        "write": 0,
    }
    assert (indices["arg"], indices["read"]) == (1, 16)
    assert (produced["arg"], produced["read"], produced["write"]) == ("result", 0, rows)


def test_an_evaluator_that_misses_an_operand_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(
        cost_evaluator_registry, "lookup", lambda cls: lambda call, ctx: Cost({}, ())
    )

    with pytest.raises(AnalysisError, match="op=Binary: cost reports 0 operands, the call has 3"):
        _run(_wide_grid, "compute-cost")


def test_memory_holds_parameters_resident_past_their_last_reader() -> None:
    """A function cannot reclaim storage its caller handed to it."""
    entry = _WeightedAdd.entry_function()
    analyze(_WeightedAdd, entry, analysis="memory")

    record = get_metadata(entry, MemoryMetadata)
    assert record is not None
    held = [item for item in record.lifetimes if item.persistent]
    ordinary = [item for item in record.lifetimes if not item.persistent]
    assert [item.binding for item in held] == ["source", "weight"]
    assert ordinary, "body allocations are still measured by their live ranges"
    assert all(
        item.last_used_at == max(row.last_used_at for row in record.lifetimes) for item in held
    )


def test_movement_costs_follow_each_operations_materialization() -> None:
    functions = {function.name: function for function in _MovementCosts.functions}

    for name in ("row", "column"):
        function = functions[name]
        analyze(_MovementCosts, function, analysis="compute-cost")
        (move,) = _calls(function)
        record = get_metadata(move, ComputeCostMetadata)
        assert record is not None
        kept = 256 * 2048 * 4
        assert record.traffic_at("gmem") == TrafficBytes(read=kept, write=kept)

    materialized = functions["materialized"]
    analyze(_MovementCosts, materialized, analysis="memory")
    transpose, reshape = _calls(materialized)
    transpose_cost = get_metadata(transpose, ComputeCostMetadata)
    reshape_cost = get_metadata(reshape, ComputeCostMetadata)
    assert transpose_cost is not None
    moved = 1024 * 2048 * 4
    assert transpose_cost.traffic_at("gmem") == TrafficBytes(read=moved, write=moved)
    assert reshape_cost is not None
    assert reshape_cost.traffic_at("gmem").total_bytes == 0
    footprint = get_metadata(materialized, MemoryMetadata)
    assert footprint is not None
    assert any(not item.persistent and item.bytes == moved for item in footprint.lifetimes)

    copied = functions["copied"]
    analyze(_MovementCosts, copied, analysis="compute-cost")
    selected, concat = _calls(copied)
    selected_cost = get_metadata(selected, ComputeCostMetadata)
    concat_cost = get_metadata(concat, ComputeCostMetadata)
    assert selected_cost is not None
    selected_bytes = 256 * 2048 * 4
    assert selected_cost.traffic_at("gmem") == TrafficBytes(
        read=selected_bytes, write=selected_bytes
    )
    assert concat_cost is not None
    assert concat_cost.traffic_at("gmem") == TrafficBytes(
        read=2 * selected_bytes,
        write=2 * selected_bytes,
    )


def test_runtime_slice_costs_the_selected_region() -> None:
    source = Var(type=make_tensor_type((1024, 2048)), name="source")
    runtime_end = Var(type=make_tensor_type((), DType.i64), name="end")
    output = make_tensor_type((256, 2048))
    call = Call(
        type=output,
        target=Slice(begin=(0, 0), end=(runtime_end, 2048), strides=(1, 1)),
        args=(source,),
    )

    cost = CostEvaluator(CostContext(selected_output_type=output)).visit_Call(call)

    kept = 256 * 2048 * 4
    assert cost.traffic == (
        TrafficBytes(read=kept),
        TrafficBytes(write=kept),
    )


def test_a_sharded_shared_tile_fits_once_and_advises_on_its_peak() -> None:
    functions = {function.name: function for function in _SharedTile.functions}
    split = functions["split"]
    analyze(_SharedTile, split, analysis="memory")

    record = get_metadata(split, MemoryMetadata)
    assert record is not None
    smem = next(item for item in record.footprint if item.level == "smem")
    assert smem.capacity_bytes == 232_448
    assert smem.peak_bytes == 422_400
    assert max(item.bytes for item in record.lifetimes if item.level == "smem") == 211_200
    assert any(
        "smem peak is 422400 B" in note and "order-dependent" in note and "not a bound" in note
        for note in record.advisories
    )

    with pytest.raises(AnalysisError, match=r"27878400 B in smem"):
        analyze(_SharedTile, functions["broadcast"], analysis="memory")


def test_memory_footprints_follow_the_owner_recorded_by_the_target() -> None:
    matmul = next(fn for fn in _MatmulLayouts.functions if fn.name == "split")
    analyze(_MatmulLayouts, matmul, analysis="memory")
    gmem = get_metadata(matmul, MemoryMetadata)
    assert gmem is not None
    gmem_lifetimes = {item.binding: item.bytes for item in gmem.lifetimes if item.level == "gmem"}
    assert gmem_lifetimes["local_lhs"] == gmem_lifetimes["lhs"] == 4_325_376

    shared = next(fn for fn in _SharedTile.functions if fn.name == "split")
    analyze(_SharedTile, shared, analysis="memory")
    cta_owned = get_metadata(shared, MemoryMetadata)
    assert cta_owned is not None
    assert (
        next(
            item.bytes
            for item in cta_owned.lifetimes
            if item.binding == "local" and item.level == "smem"
        )
        == 211_200
    )

    thread_shared = _entry(_modest_shared)
    analyze(_modest_shared, thread_shared, analysis="memory")
    still_cta_owned = get_metadata(thread_shared, MemoryMetadata)
    assert still_cta_owned is not None
    assert (
        next(
            item.bytes
            for item in still_cta_owned.lifetimes
            if item.binding == "local" and item.level == "smem"
        )
        == 4_096
    )

    registers = _entry(_thread_sharded)
    analyze(_thread_sharded, registers, analysis="memory")
    thread_owned = get_metadata(registers, MemoryMetadata)
    assert thread_owned is not None
    assert (
        next(
            item.bytes
            for item in thread_owned.lifetimes
            if item.binding == "local" and item.level == "rmem"
        )
        == 8
    )


def test_a_cache_too_small_is_advisory_and_only_where_the_scopes_agree() -> None:
    """An over-full cache costs speed, so the analysis still succeeds.

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
    """The sharing edge makes L1's usable size depend on the program.

    The sharing edge makes L1's usable size depend on the program, and that
    division is reportable without any working set.
    """
    _, entry = _run(_modest_shared, "memory")

    record = get_metadata(entry, MemoryMetadata)
    assert record is not None
    block = CudaTarget("nvidia.h200_sxm").architecture.unified_l1_shared_per_sm_bytes
    smem = record.level("smem")
    assert smem is not None and smem.peak_bytes > 0
    assert any(
        f"smem claims {smem.peak_bytes} B of the {block} B block" in note
        and f"leaving l1 {block - smem.peak_bytes} B" in note
        for note in record.advisories
    )


def test_roofline_reads_the_recorded_work_and_aggregates_before_dividing() -> None:
    """The bound follows whatever compute-cost recorded, at the target's rate.

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

    facts = CudaTarget("nvidia.h200_sxm").get_facts(ThroughputFacts)
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
    assert whole.ideal_ns == max(whole.compute_ns, whole.memory_ns)


def test_timeline_credits_an_unplaced_call_with_one_position() -> None:
    """A declared launch hierarchy does not place a value by itself."""
    _, entry = _run(_wide_grid, "timeline")

    record = get_metadata(_calls(entry)[-1], TimelineMetadata)
    assert record is not None
    assert record.grid_units == 1
    assert record.waves == 1

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
    """L1 caches L2, L2 caches GMEM, and L1 shares a block with SMEM.

    The shared edge is what makes the graph non-hierarchical: smem is an
    addressable level and l1 a cache, yet neither contains the other. A machine
    with no such block describes a plain cache chain in the same shape rather
    than carrying a stub edge, which is the case that would otherwise be modelled
    by a special one.
    """
    facts = CudaTarget("nvidia.h200_sxm").get_facts(MemoryHierarchyFacts)

    assert facts.cached_level("l1") == "l2"
    assert facts.cached_level("l2") == "gmem"
    assert facts.backing_level("l1") == "gmem"
    assert facts.capacity_sharers("l1") == (
        ("smem", CudaTarget("nvidia.h200_sxm").architecture.unified_l1_shared_per_sm_bytes),
    )
    assert {level.name for level in facts.explicit_levels} >= {"gmem", "smem", "rmem"}
    assert {level.name for level in facts.implicit_levels} == {"l1", "l2"}

    flat = AmxTarget().get_facts(MemoryHierarchyFacts)
    assert flat.cached_level("l1d") == "l2"
    assert flat.backing_level("l1d") == "gmem"
    assert flat.capacity_sharers("l1d") == ()
    assert all(relation.kind is MemoryRelationKind.CACHES for relation in flat.relations)


def test_a_report_shows_the_requested_analyses_and_reads_the_same_either_way() -> None:
    """Promote only bounded dependency evidence into a roofline report.

    A requested root pulls its dependencies in, so their records land on the
    IR without having been asked for. Roofline promotes only the bounded
    whole-program evidence that explains its verdict; dependency details stay on
    the IR for whoever does ask.

    Two formats over one structure cannot disagree about a conclusion, so the
    text is checked to carry the verdict the JSON states.
    """
    entry = _entry(_CudaAdd)
    result = analyze(_CudaAdd, entry, analysis="roofline")

    data = report([result])

    assert data["requested"] == ["roofline"]
    assert data["executed"] == ["compute-cost", "memory", "roofline"]
    assert set(data["function_records"]) == {"memory", "roofline"}
    memory = data["function_records"]["memory"]
    assert set(memory) == {"footprint"}
    assert all(set(item) == {"level", "peak_bytes"} for item in memory["footprint"])
    assert data["totals"]["flops"] == {"f32": 256}
    assert all(set(call) == {"value", "roofline"} for call in data["calls"])
    assert get_metadata(entry, MemoryMetadata) is not None
    cost = get_metadata(_calls(entry)[-1], ComputeCostMetadata)
    assert cost is not None
    assert " operands=0:r" in cost.format_comment()
    assert ",result:r" in cost.format_comment()

    payload = json.loads(render_json(data))
    assert payload == data
    text = render_text(data)
    assert f"by={payload['function_records']['roofline']['bound_by']}" in text
    assert "# flops " in text
    assert "# traffic " in text
    assert "# peak-footprint " in text
    assert "operands" not in text
