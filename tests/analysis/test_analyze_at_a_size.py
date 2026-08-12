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

from tests.fixtures.placed.gqa_decode import MAX_CTX, GqaOnline
from tests.fixtures.placed.prefill_decode_attention import PrefillDecodeAttention
from tests.models.qwen3_1_7b.case import CASE as QWEN3_1_7B
from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    analyze,
)
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.walk import enclosing_trips, postorder
from tilefoundry.inspection.analysis_report import render_text, report
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.specialize import (
    display_name,
    origin_of,
    residual_dims,
    variant_for,
)
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule import ScheduleError, ScheduleOptions, schedule
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


@pytest.mark.parametrize("family", FAMILIES)
def test_every_analysis_runs_at_a_stated_size(family: str) -> None:
    module = _aimed()

    result = analyze(module, module.entry_function(), analysis=family, dims=DIMS)

    assert result.metadata_types
    assert result.module is module


@pytest.mark.parametrize(
    ("dims", "variant", "bound_by", "ideal_ns"),
    [
        ({"ctx": 1024, "seq": 1}, "decode", "memory", 3_496),
        ({"ctx": 1, "seq": 1024}, "prefill", "compute", 65_449),
    ],
    ids=["decode-open-context", "prefill-open-sequence"],
)
def test_block_attention_selects_and_analyzes_each_placed_regime(
    dims: dict[str, int], variant: str, bound_by: str, ideal_ns: int
) -> None:
    result = analyze(
        PrefillDecodeAttention,
        PrefillDecodeAttention.entry_function(),
        analysis="roofline",
        dims=dims,
    )

    assert display_name(origin_of(result.function)) == variant
    record = get_metadata(result.function, RooflineMetadata)
    assert record is not None
    assert record.ideal_ns == ideal_ns
    assert record.bound_by == bound_by


@pytest.mark.parametrize("family", FAMILIES)
def test_the_result_names_the_function_that_carries_the_records(family: str) -> None:
    """The records are written onto the program that was measured, and that is the derived one.

    The records are written onto the program that was measured, and that is
    the derived one. Handing back the symbolic input would send a reader looking
    for records on a function that has none.
    """
    module = _aimed()
    authored = module.entry_function()

    result = analyze(module, authored, analysis=family, dims=DIMS)

    assert result.function is not authored
    assert result.function.name == authored.name
    assert residual_dims(result.function) == ()


def test_a_report_at_a_size_carries_every_family_it_ran() -> None:
    """Several requested roots record all conclusions on one concrete view."""
    module = _aimed()
    authored = module.entry_function()

    result = analyze(module, authored, analysis=FAMILIES, dims=DIMS)
    data = report(result)

    assert result.analyses == FAMILIES
    assert data["executed"] == list(FAMILIES)
    assert set(data["function_records"]) == {
        "compute-cost",
        "memory",
        "roofline",
        "timeline",
    }
    assert data["totals"]["flops"], "the work totals summed to nothing"
    text = render_text(data)
    for expected in ("peak-footprint", "ideal-bound", "theoretical-makespan"):
        assert expected in text, f"{expected} is missing from the rendered report"

    single_module = _aimed()
    single = analyze(
        single_module,
        single_module.entry_function(),
        analysis="timeline",
        dims=DIMS,
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


def test_a_report_at_a_size_carries_the_per_call_records_of_every_family() -> None:
    """Per-Call records from several roots inhabit the same occurrences."""
    module = _aimed()
    authored = module.entry_function()

    result = analyze(
        module,
        authored,
        analysis=("compute-cost", "timeline"),
        dims=DIMS,
    )
    data = report(result)

    families = {name for row in data["calls"] for name in row if name != "value"}
    assert families == {"compute-cost", "timeline"}, families


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


@pytest.mark.parametrize(
    "ctx_len",
    (1, 1024),
)
def test_qwen_decoder_unplaced_calls_have_one_position_at_each_sequence_length(
    ctx_len: int,
) -> None:
    module = QWEN3_1_7B.build()
    function = module.lookup("decoder_layer")

    result = analyze(module, function, analysis="timeline", dims={"ctx_len": ctx_len})

    record = get_metadata(result.function, TimelineMetadata)
    assert record is not None
    assert record.grid_units == 1
    assert record.waves == 83
    assert record.end_ns == 10_537


def test_gqa_loop_occurrences_cost_one_trip_while_the_root_applies_all_trips() -> None:
    results = []
    for extent in (8, 16):
        module = _aimed()
        results.append(
            analyze(
                module,
                module.entry_function(),
                analysis="timeline",
                dims={"ctx_len": extent},
            )
        )

    loop_casts = []
    for result, extent in zip(results, (8, 16)):
        trips = enclosing_trips(result.function.body)
        records = []
        for expr in postorder(result.function.body):
            if not (
                isinstance(expr, Call)
                and isinstance(expr.target, Cast)
                and trips.get(id(expr)) == extent
            ):
                continue
            record = get_metadata(expr, ComputeCostMetadata)
            assert record is not None
            records.append(record)
        assert len(records) == 2
        loop_casts.append(tuple(records))

    assert loop_casts[0] == loop_casts[1]
    assert [record.flops for record in loop_casts[0]] == [(("f32", 4096),)] * 2

    roots = []
    for result in results:
        root = get_metadata(result.function, ComputeCostMetadata)
        assert root is not None
        roots.append(root)
    assert [dict(root.flops)["f32"] for root in roots] == [276_704, 508_128]

    loop_timelines = []
    for result, extent in zip(results, (8, 16)):
        trips = enclosing_trips(result.function.body)
        records = tuple(
            get_metadata(expr, TimelineMetadata)
            for expr in postorder(result.function.body)
            if isinstance(expr, Call)
            and isinstance(expr.target, Cast)
            and trips.get(id(expr)) == extent
        )
        assert all(record is not None for record in records)
        loop_timelines.append(records)
    assert loop_timelines[0] == loop_timelines[1]

    timeline_roots = [
        get_metadata(result.function, TimelineMetadata) for result in results
    ]
    assert all(root is not None for root in timeline_roots)
    assert [root.end_ns for root in timeline_roots] == [311, 619]


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
