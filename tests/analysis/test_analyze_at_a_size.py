"""Analysing and scheduling a function authored for a range of sizes.

An analysis counts elements and holds them against a machine; a solver lays
work across a level by counting it. Neither has an answer for a dimension that
is still a range, so the size is stated at the call and the program that gets
measured is the one at that size.

What the call accepts stays narrow: a function this Module owns. Choosing the
size happens after that, so nothing here widens which programs a Module will
answer for -- it only lets the ones it owns be asked about at a size.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.fixtures.placed.flash_split_k_decode import BLOCK, HEADS, WORKERS, FlashSplitKDecode
from tests.fixtures.placed.gqa_decode import MAX_CTX, GqaOnline
from tests.fixtures.placed.prefill_decode_attention import PrefillDecodeAttention
from tests.models.qwen3_1_7b.case import CASE as QWEN3_1_7B
from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TimelineSummaryMetadata,
    analyze,
)
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.walk import enclosing_trips, postorder, tensor_types
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.ir.core import Call, Constant, get_metadata
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.specialize import (
    display_name,
    origin_of,
    residual_dims,
    variant_for,
)
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.index_select import IndexSelect
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types import tensor_bytes
from tilefoundry.ir.types.shard import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    Topology,
    shard_layout_of,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.schedule import ScheduleError, ScheduleOptions, schedule
from tilefoundry.schedule.partition import build_partition_program
from tilefoundry.target import CudaTarget

CONTEXT = 32
DIMS = {"ctx_len": CONTEXT}
FAMILIES = ("compute-cost", "memory", "roofline", "timeline")


SOLVER = ScheduleOptions(timeout_seconds=60, workers=4, random_seed=0, stop_at_first_solution=True)


def _aimed():
    """The decode example, aimed at one machine."""
    return replace(
        GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)
    )


def _subject(family: str):
    """A concrete query that satisfies the selected family's readiness."""
    module = _aimed()
    return module, module.entry_function(), DIMS


@pytest.mark.parametrize("family", FAMILIES)
def test_every_analysis_runs_at_a_stated_size(family: str) -> None:
    module, function, dims = _subject(family)

    result = analyze(module, function, analysis=family, dims=dims)

    assert result.metadata_types
    assert result.module is module


@pytest.mark.parametrize(
    ("small_dims", "large_dims", "variant", "bound_by", "min_scale", "max_scale"),
    [
        (
            {"ctx": 512, "seq": 1},
            {"ctx": 1024, "seq": 1},
            "decode",
            "memory",
            1.9,
            2.1,
        ),
        (
            {"ctx": 1, "seq": 512},
            {"ctx": 1, "seq": 1024},
            "prefill",
            "compute",
            3.8,
            4.2,
        ),
    ],
    ids=["decode-open-context", "prefill-open-sequence"],
)
def test_block_attention_selects_and_analyzes_each_placed_regime(
    small_dims: dict[str, int],
    large_dims: dict[str, int],
    variant: str,
    bound_by: str,
    min_scale: float,
    max_scale: float,
) -> None:
    records = []
    for dims in (small_dims, large_dims):
        result = analyze(
            PrefillDecodeAttention,
            PrefillDecodeAttention.entry_function(),
            analysis="roofline",
            dims=dims,
        )
        concrete = origin_of(result.function)
        assert concrete is not None
        selected = origin_of(concrete)
        assert selected is not None
        assert display_name(selected) == variant
        record = get_metadata(result.function, RooflineMetadata)
        assert record is not None
        assert record.bound_by == bound_by
        records.append(record)

    ideal_scale = records[1].ideal_ns / records[0].ideal_ns
    assert min_scale < ideal_scale < max_scale


def test_split_k_decode_analyzes_each_offset_window_at_ctx_4096() -> None:
    result = analyze(
        FlashSplitKDecode,
        FlashSplitKDecode.entry_function(),
        analysis="roofline",
        dims={"ctx": 4096},
    )

    assert result.function is not FlashSplitKDecode.entry_function()
    loops = [
        expr for expr in postorder(result.function.body) if isinstance(expr, GridRegionExpr)
    ]
    assert len(loops) == 1
    (loop,) = loops
    assert loop.step == BLOCK * WORKERS

    loop_exprs = postorder(loop)
    assert not any(
        isinstance(attr, Partial)
        for expr in loop_exprs
        if isinstance(expr, Call)
        for tensor in tensor_types(expr.type)
        for layout in (shard_layout_of(tensor.layout),)
        if isinstance(layout, ShardLayout)
        for attr in layout.attrs
    )

    slices = [
        expr
        for expr in loop_exprs
        if isinstance(expr, Call) and isinstance(expr.target, Slice)
    ]
    assert len(slices) == 2
    for window in slices:
        starts = window.args[1].elements
        base = starts[1]
        assert isinstance(base, Call)
        assert base.args[0] is loop.induction_var
        assert isinstance(base.target, Binary) and base.target.kind is BinaryKind.ADD
        offset = base.args[1]
        assert isinstance(offset, Call)
        assert isinstance(offset.target, Binary) and offset.target.kind is BinaryKind.MUL
        assert isinstance(offset.args[1], Constant) and offset.args[1].value == BLOCK

        worker_index = offset.args[0]
        assert isinstance(worker_index, Call)
        assert isinstance(worker_index.target, Reshape)
        local_index = worker_index.args[0]
        assert isinstance(local_index, Call)
        assert isinstance(local_index.target, Local)
        worker_shard = local_index.args[0]
        assert isinstance(worker_shard, Call)
        assert isinstance(worker_shard.target, Reshard)
        worker_source = worker_shard.args[0]
        assert isinstance(worker_source, Call)
        assert isinstance(worker_source.target, Arange)
        assert worker_source.target.end == WORKERS
        assert worker_source.target.start == 0
        assert worker_shard.target.storage is StorageKind.RMEM
        worker_layout = worker_shard.target.layout
        assert isinstance(worker_layout, ShardLayout)
        assert worker_layout.mesh.layout.shape == (HEADS, WORKERS)
        assert worker_layout.attrs == (Broadcast(), Split(axis=0))
        assert window.target.sizes[1] == BLOCK
        record = get_metadata(window, ComputeCostMetadata)
        assert record is not None
        assert record.traffic_at("rmem").read > 0

    first_offset = slices[0].args[1].elements[1].args[1]
    second_offset = slices[1].args[1].elements[1].args[1]
    first_worker_index = first_offset.args[0]
    second_worker_index = second_offset.args[0]
    assert first_worker_index is second_worker_index

    cache_bytes = tensor_bytes(result.function.params[1].type)
    kv_windows = [
        expr
        for expr in loop_exprs
        if isinstance(expr, Call)
        and isinstance(expr.target, Reshard)
        and expr.args
        and isinstance(expr.args[0], Call)
        and isinstance(expr.args[0].target, Slice)
    ]
    assert len(kv_windows) == 2
    trips = enclosing_trips(result.function.body)
    assert all(
        trips.get(id(window), 1)
        * get_metadata(window, ComputeCostMetadata).traffic_at("gmem").read
        == cache_bytes // WORKERS
        for window in kv_windows
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_the_result_names_the_function_that_carries_the_records(family: str) -> None:
    """The records are written onto the program that was measured, and that is the derived one.

    The records are written onto the program that was measured, and that is
    the derived one. Handing back the symbolic input would send a reader looking
    for records on a function that has none.
    """
    module, authored, dims = _subject(family)

    result = analyze(module, authored, analysis=family, dims=dims)

    assert result.function is not authored
    assert result.function.name == authored.name
    assert residual_dims(result.function) == ()


def test_one_root_and_four_roots_produce_the_same_timeline_records() -> None:
    """A union closure preserves the conclusion of its independently requested root."""
    module, authored, dims = _subject("timeline")

    result = analyze(module, authored, analysis=FAMILIES, dims=dims)

    assert result.analyses == FAMILIES
    single_module, single_function, single_dims = _subject("timeline")
    single = analyze(
        single_module,
        single_function,
        analysis="timeline",
        dims=single_dims,
    )
    assert tuple(
        get_metadata(expr, TimelineMetadata)
        for expr in (single.function, *postorder(single.function.body))
        if get_metadata(expr, TimelineMetadata) is not None
    ) == tuple(
        get_metadata(expr, TimelineMetadata)
        for expr in (result.function, *postorder(result.function.body))
        if get_metadata(expr, TimelineMetadata) is not None
    )
    assert get_metadata(single.function, TimelineSummaryMetadata) == get_metadata(
        result.function, TimelineSummaryMetadata
    )


def test_without_a_size_the_result_names_the_record_bearing_view() -> None:
    """A static input remains authored while its analysis view carries records."""
    module = QWEN3_1_7B.build()
    function = module.lookup("mlp")

    result = analyze(module, function, analysis="compute-cost")

    assert result.module is module
    assert result.function.name == function.name
    assert origin_of(result.function) is function
    assert get_metadata(result.function, ComputeCostMetadata) is not None
    assert get_metadata(function, ComputeCostMetadata) is None


def test_gqa_loop_occurrences_are_costed_once_and_parameterized_over_trips() -> None:
    results = []
    for extent in (8, 16):
        module = _aimed()
        results.append(
            analyze(
                module,
                module.entry_function(),
                analysis=("compute-cost", "timeline"),
                dims={"ctx_len": extent},
            )
        )

    loop_casts = []
    loop_timelines = []
    root_timelines = []
    for result, extent in zip(results, (8, 16)):
        trips = enclosing_trips(result.function.body)
        costs = []
        timelines = []
        structural = []
        for expr in postorder(result.function.body):
            if not isinstance(expr, Call) or trips.get(id(expr)) != extent:
                continue
            timeline = get_metadata(expr, TimelineMetadata)
            assert timeline is not None
            timelines.append(timeline)
            if isinstance(expr.target, Reshape):
                cost = get_metadata(expr, ComputeCostMetadata)
                assert cost is not None
                structural.append((cost, timeline))
            if isinstance(expr.target, Cast):
                cost = get_metadata(expr, ComputeCostMetadata)
                assert cost is not None
                costs.append(cost)
        assert len(costs) == 2
        assert len(timelines) == 18
        assert len(structural) == 1
        assert {record.trips for record in timelines} == {extent}
        assert {record.stride_ns for record in timelines} == {920}
        structural_cost, structural_timeline = structural[0]
        assert structural_cost.flops_per_unit == ()
        assert structural_cost.traffic_per_unit_at("gmem").total_bytes == 0
        assert structural_cost.traffic_per_unit == ()
        assert (
            structural_timeline.start_ns,
            structural_timeline.end_ns,
        ) == (652, 652)
        loop_casts.append(tuple(costs))
        loop_timelines.append(tuple(timelines))

        loop = next(
            expr
            for expr in postorder(result.function.body)
            if isinstance(expr, GridRegionExpr)
        )
        consumers = [
            expr
            for expr in postorder(result.function.body)
            if isinstance(expr, Call) and any(arg is loop for arg in expr.args)
        ]
        loop_start = min(record.start_ns for record in timelines)
        loop_end = loop_start + extent * timelines[0].stride_ns
        assert len(consumers) == 3
        assert {
            get_metadata(consumer, TimelineMetadata).start_ns
            for consumer in consumers
        } == {loop_end}

        root_timeline = get_metadata(result.function, TimelineSummaryMetadata)
        assert root_timeline is not None
        root_timelines.append(root_timeline)

    assert loop_casts[0] == loop_casts[1]
    assert [record.flops for record in loop_casts[0]] == [(("f32", 512),)] * 2
    assert [
        tuple(
            (
                record.start_ns,
                record.end_ns,
                record.stride_ns,
            )
            for record in records
        )
        for records in loop_timelines
    ] == [
        tuple(
            (
                record.start_ns,
                record.end_ns,
                record.stride_ns,
            )
            for record in loop_timelines[0]
        )
    ] * 2
    assert (loop_timelines[0][0].start_ns, loop_timelines[0][0].end_ns) == (
        652,
        652,
    )
    assert [record.local_makespan_ns for record in root_timelines] == [9_129, 16_489]
    assert [record.waves for record in root_timelines] == [1, 1]
    assert [record.estimated_kernel_ns for record in root_timelines] == [9_129, 16_489]

    roots = []
    for result in results:
        root = get_metadata(result.function, ComputeCostMetadata)
        assert root is not None
        roots.append(root)
    assert [dict(root.flops)["f32"] for root in roots] == [211_936, 385_760]

    rendered = render_analysis(results[0])
    lines = rendered.annotated.splitlines()
    rows = [row for row in rendered.data["calls"] if "timeline" in row]
    comments = [
        line.split("; timeline ", 1)[1]
        for line in lines
        if "; timeline " in line
    ]
    expected_comments = []
    for row in rows:
        value, line_text = row["value"].rsplit(":", 1)
        line = int(line_text)
        assert lines[line - 1].lstrip().startswith(f"{value} = ")
        timeline = row["timeline"]
        if timeline["trips"] > 1:
            span_end = (
                timeline["start_ns"]
                + timeline["trips"] * timeline["stride_ns"]
            )
            expected_comments.append(
                f"[{timeline['start_ns']}+{timeline['stride_ns']}t, "
                f"{timeline['end_ns']}+{timeline['stride_ns']}t) "
                f"trips={timeline['trips']} "
                f"span=[{timeline['start_ns']},{span_end})"
            )
        else:
            expected_comments.append(
                f"start={timeline['start_ns']}ns end={timeline['end_ns']}ns"
            )

    assert comments == expected_comments
    first = next(row for row in rows if row["value"].startswith("v0:"))
    assert first["value"] == "v0:44"
    assert lines[43].lstrip().startswith("v0 = reshard(")
    assert "; timeline start=0ns end=0ns" in lines[47]
    structural = next(row for row in rows if row["value"].startswith("v10:"))
    assert set(structural) == {"value", "compute-cost", "timeline"}
    assert structural["value"] == "v10:59"
    assert structural["timeline"] == {
        "end_ns": 652,
        "start_ns": 652,
        "stride_ns": 920,
        "trips": 8,
    }
    assert "timeline [652+920t, 652+920t) trips=8 span=[652,8012)" in lines[58]
    assert lines[82].strip() == "m = v16"
    assert "timeline" not in lines[82]
@pytest.mark.parametrize("ctx_len", (1, 1024))
def test_qwen_decoder_unplaced_calls_are_refused_at_each_sequence_length(
    ctx_len: int,
) -> None:
    module = QWEN3_1_7B.build()
    function = module.lookup("decoder_layer")

    with pytest.raises(
        AnalysisError,
        match=r"model.py:\d+:.*has no cta placement",
    ):
        analyze(module, function, analysis="timeline", dims={"ctx_len": ctx_len})


def test_qwen_decoder_keeps_rotary_and_kv_cache_parameters_resident() -> None:
    module = QWEN3_1_7B.build()
    function = module.lookup("decoder_layer")

    result = analyze(module, function, analysis="memory", dims={"ctx_len": 1024})

    record = get_metadata(result.function, MemoryMetadata)
    assert record is not None
    lifetimes = {item.binding: item for item in record.lifetimes}
    cache_names = ("cos_cache", "sin_cache", "k_cache", "v_cache")
    assert all(lifetimes[name].persistent for name in cache_names)


def test_a_size_no_variant_covers_is_refused() -> None:
    module = _aimed()

    with pytest.raises(AnalysisError, match="no variant covering"):
        analyze(
            module,
            module.entry_function(),
            analysis="compute-cost",
            dims={"ctx_len": MAX_CTX},
        )


def test_a_dimension_the_function_does_not_have_is_refused() -> None:
    module = _aimed()

    with pytest.raises(AnalysisError, match="no dimension named"):
        analyze(
            module,
            module.entry_function(),
            analysis="compute-cost",
            dims={**DIMS, "batch": 2},
        )


def test_a_dimension_left_unbound_is_refused() -> None:
    """Test a dimension left unbound is refused.

    Stating some other dimension is useful while the choices are being made
    and useless to an analysis, which would meet the unbound one as an extent
    that is not a number.
    """
    module = _aimed()

    with pytest.raises(AnalysisError, match="was not given a size"):
        analyze(
            module,
            module.entry_function(),
            analysis="compute-cost",
            dims={"batch": 4},
        )


def test_an_empty_or_malformed_size_is_refused_rather_than_ignored() -> None:
    """A caller who believes they stated a size must not be left believing it."""
    module = _aimed()
    entry = module.entry_function()

    with pytest.raises(AnalysisError, match="non-empty mapping"):
        analyze(module, entry, analysis="compute-cost", dims={})
    with pytest.raises(AnalysisError, match="takes an integer extent"):
        analyze(module, entry, analysis="compute-cost", dims={"ctx_len": 32.0})


def test_a_size_states_nothing_about_a_function_from_elsewhere() -> None:
    """Ownership is settled before a size is looked at.

    Ownership is settled before a size is looked at, so a foreign function
    is refused for being foreign rather than for its dimensions.
    """
    module = _aimed()
    foreign = QWEN3_1_7B.build().lookup("mlp")

    with pytest.raises(AnalysisError, match="is not a function of module"):
        analyze(module, foreign, analysis="compute-cost", dims=DIMS)


def test_scheduling_at_a_stated_size_plans_and_verifies() -> None:
    """The plan is a plan for one size, and it is checked against the program of that size.

    The plan is a plan for one size, and it is checked against the program of
    that size -- which is the one the result names.
    """
    module = _aimed()
    authored = module.entry_function()

    result = schedule(module, authored, topology="cta", options=SOLVER, dims=DIMS)

    assert result.module is module
    assert result.function is not authored
    assert residual_dims(result.function) == ()
    result.plan.verify(module, result.function, result.topology)
    assert result.plan.to_json() == result.plan.to_json()

    program = build_partition_program(module, result.function)
    selections = [site for site in program.sites if isinstance(site.call.target, IndexSelect)]
    assert len(selections) == 2
    assert {
        program.values[site.input_value_ids[0][0]].source.name
        for site in selections
    } == {"k_cache", "v_cache"}


def test_scheduling_refuses_a_size_no_variant_covers() -> None:
    module = _aimed()

    with pytest.raises(ScheduleError, match="no variant covering"):
        schedule(
            module,
            module.entry_function(),
            topology="cta",
            options=SOLVER,
            dims={"ctx_len": MAX_CTX},
        )


def test_the_entry_at_a_chosen_size_is_still_the_entry() -> None:
    """The device-wide solver admits only the entry, and it decides that by name.

    The device-wide solver admits only the entry, and it decides that by
    name: a function specialised from the entry is a different object and the
    same program.
    """
    module = _aimed()
    variant = variant_for(module.entry_function(), DIMS)

    assert variant.name == module.entry_function().name
    with pytest.raises(ScheduleError, match="requires the module entry"):
        schedule(
            module,
            module.lookup("_ctx_partials"),
            topology="cta",
            options=SOLVER,
            dims=DIMS,
        )
