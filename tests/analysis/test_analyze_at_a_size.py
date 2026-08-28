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

from tests.fixtures.placed.gqa_decode import GqaOnline
from tests.fixtures.placed.mha_decode_paged import LongerCache, ShorterCache
from tests.fixtures.placed.qwen3_1_7b_pd import PrefillLayer
from tests.models.corpus import ConcreteCase, placed_cases
from tests.models.qwen3_1_7b.case import CASE as QWEN3_1_7B
from tilefoundry.analysis import (
    AnalysisResult,
    ComputeCostMetadata,
    LoopFootprintMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RooflineMetadata,
    TrafficMetadata,
    analyze,
)
from tilefoundry.analysis.compute_cost import _local_duration_ns
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.scope import build_scopes, walk_scopes
from tilefoundry.ir.core import Call, describe_expr, get_metadata
from tilefoundry.ir.core.metadata import ExecutionDomainMetadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.specialize import (
    origin_of,
    residual_dims,
    variant_for,
)
from tilefoundry.ir.types.shard import (
    Topology,
)
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.schedule import ScheduleError, ScheduleOptions, schedule
from tilefoundry.target import CudaTarget, PerformanceServiceFacts, ThroughputFacts

CONTEXT = 32
DIMS = {"ctx_len": CONTEXT}
FAMILIES = ("compute-cost", "memory", "roofline", "performance")
INVENTORY = [pytest.param(case, id=case.id) for case in placed_cases()]


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


def assert_performance_contract(result: AnalysisResult) -> None:
    """Every performance conclusion traces back to what it was derived from.

    The prediction contains each occurrence it timed and is no faster than the
    ideal bound. An occurrence's duration is its own compute-cost record priced
    at the target's rates, and a solve that proved nothing says so. One a loop
    repeats is written once, so its interval is that many of its own durations
    and its last trip still lands inside the prediction that contains it.
    A loop is not an occurrence and carries no timeline of its own, and still
    states the buffers it touches.
    """
    fn = result.function
    summary = get_metadata(fn, PerformanceSummaryMetadata)
    assert summary is not None
    assert 0 <= summary.timeline.start_ns <= summary.timeline.end_ns
    placement = get_metadata(fn, MemoryMetadata)
    assert placement is not None and placement.allocation is not None
    assert placement.allocation.solver_status in ("optimal", "feasible")
    predicted_ns = summary.timeline.end_ns - summary.timeline.start_ns
    assert summary.waves > 0 and predicted_ns % summary.waves == 0
    bound = get_metadata(fn, RooflineMetadata)
    assert bound is not None and bound.ideal_ns <= predicted_ns

    module_target = result.module.resolve_target()
    throughput = module_target.get_facts(ThroughputFacts)
    services = module_target.get_facts(PerformanceServiceFacts)
    scopes = tuple(walk_scopes(build_scopes(result.module, fn)))
    timed = 0
    for expr in collect_exprs(fn.body):
        if not isinstance(expr, Call) or isinstance(expr.target, Function):
            continue
        cost = get_metadata(expr, ComputeCostMetadata)
        assert cost is not None
        duration = _local_duration_ns(
            cost,
            throughput,
            services,
            moved=get_metadata(expr, TrafficMetadata),
            level=result.level,
        )
        record = get_metadata(expr, PerformanceMetadata)
        if not duration:
            assert record is None
            continue
        timed += 1
        assert record is not None
        assert summary.timeline.start_ns <= record.timeline.start_ns
        assert record.timeline.end_ns <= summary.timeline.end_ns

        span = record.timeline.end_ns - record.timeline.start_ns
        assert span % duration == 0, describe_expr(expr)
        runs = span // duration
        available = 1
        owner = next(
            (
                scope
                for scope in scopes
                if any(item is expr for item in scope.accesses.get("narrow", {}))
            ),
            None,
        )
        if owner is not None:
            cursor = owner
            while cursor.parent is not None:
                if cursor.is_variant(expr):
                    available *= max(1, cursor.trips())
                cursor = cursor.parent
        assert 1 <= runs <= available and available % runs == 0, describe_expr(expr)
        trips, stride = record.timeline.trips, record.timeline.stride_ns
        assert 1 <= trips <= available and available % trips == 0, describe_expr(expr)
        assert (stride == 0) if trips == 1 else (stride >= span), describe_expr(expr)
        assert (
            record.timeline.end_ns + (trips - 1) * stride <= summary.timeline.end_ns
        ), describe_expr(expr)
    assert bool(timed) is bool(predicted_ns)
    _every_number_counts_something(result)
    for expr in collect_exprs(fn.body):
        if not isinstance(expr, GridRegionExpr):
            continue
        assert get_metadata(expr, PerformanceMetadata) is None, describe_expr(expr)
        assert get_metadata(expr, PerformanceSummaryMetadata) is None, describe_expr(expr)
        assert get_metadata(expr, LoopFootprintMetadata) is not None, describe_expr(expr)


@pytest.mark.parametrize(
    ("smaller", "larger"),
    (
        ((ShorterCache, None), (LongerCache, None)),
        (
            (PrefillLayer, {"ctx_len": 512, "seq": 1}),
            (PrefillLayer, {"ctx_len": 4608, "seq": 1}),
        ),
        (
            (PrefillLayer, {"ctx_len": 0, "seq": 512}),
            (PrefillLayer, {"ctx_len": 512, "seq": 512}),
        ),
    ),
    ids=("paged-kv", "qwen-decode-history", "qwen-prefill-history"),
)
def test_more_of_the_same_work_is_never_predicted_to_take_less_time(
    smaller, larger
) -> None:
    """A longer cache and a longer history are more of the same program.

    Nothing here says how much longer the prediction should be: a model that got
    the direction wrong would be reporting that reading twice the cache costs
    less than reading half of it, which is the one comparison a reader makes
    without being told to.
    """
    assert _predicted_ns(*smaller) <= _predicted_ns(*larger)


def _every_number_counts_something(result: AnalysisResult) -> None:
    """Every quantity these four families report is a count, so none is below zero.

    Work, bytes, a footprint and a bound are all counts of something that
    happened or has to happen. A negative one is not a small answer but a
    derivation that ran backwards -- a projection dividing what it should have
    multiplied, or a difference taken the wrong way round -- and it would then be
    added into a total that still looks plausible.
    """
    fn = result.function
    for expr in (fn, *collect_exprs(fn.body)):
        for record, rows in (
            (ComputeCostMetadata, ("flops", "flops_per_unit", "service", "service_per_unit")),
            (TrafficMetadata, ()),
            (MemoryMetadata, ()),
            (RooflineMetadata, ()),
            (PerformanceMetadata, ()),
        ):
            held = get_metadata(expr, record)
            if held is None:
                continue
            for field in rows:
                for name, value in getattr(held, field):
                    assert value >= 0, f"{describe_expr(expr)}: {field}[{name}] = {value}"
            if record is TrafficMetadata:
                for field in ("whole", "per_unit"):
                    for level, moved in getattr(held, field):
                        assert moved.read >= 0 and moved.write >= 0, (
                            f"{describe_expr(expr)}: {field}[{level}] = {moved}"
                        )
                for position, moved in enumerate(held.operands):
                    assert moved.read >= 0 and moved.write >= 0, (
                        f"{describe_expr(expr)}: operand {position} = {moved}"
                    )
            if record is MemoryMetadata:
                for level in held.footprint:
                    assert level.peak_bytes >= 0 and level.persistent_bytes >= 0
                for item in held.lifetimes:
                    assert item.bytes >= 0 and 0 <= item.defined_at <= item.last_used_at
            if record is RooflineMetadata:
                assert held.ideal_ns >= 0 and held.compute_ns >= 0 and held.memory_ns >= 0
            if record is PerformanceMetadata:
                assert 0 <= held.timeline.start_ns <= held.timeline.end_ns
    for expr in collect_exprs(fn.body):
        record = get_metadata(expr, LoopFootprintMetadata)
        if record is None:
            continue
        for item in record.footprints:
            assert item.bytes >= 0 and item.device_bytes >= 0 and item.repeated_bytes >= 0


@pytest.mark.parametrize("case", INVENTORY)
def test_every_concrete_program_answers_for_where_it_runs(case: ConcreteCase) -> None:
    """Every placed program, at every size and selector it exposes.

    This inventory is the whole of what these four analyses are held to: it is
    read off the directory rather than from a list beside it, so a program added
    there is asked the same questions without anyone choosing to ask. Each of
    them runs something inside a CTA Mesh, so each is asked for all four
    families and has to answer with a coherent prediction.
    """
    owner, function = case.program()
    result = analyze(owner, function, analysis=FAMILIES, dims=case.dims)

    assert result.module is owner
    assert set(result.executed) == set(FAMILIES)
    undomained = [
        describe_expr(call)
        for call in collect_exprs(result.function.body)
        if isinstance(call, Call)
        and not isinstance(call.target, Function)
        and (get_metadata(call, ExecutionDomainMetadata) or ExecutionDomainMetadata())
        .at("cta")
        is None
    ]
    assert not undomained, f"{case.id}: {undomained}"
    assert_performance_contract(result)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_analysis_runs_at_a_stated_size(family: str) -> None:
    module, function, dims = _subject(family)

    result = analyze(module, function, analysis=family, dims=dims)

    assert result.metadata_types
    assert result.module is module


def _predicted_ns(module, dims=None) -> int:
    """What the four families together say one program takes."""
    result = analyze(
        module, module.entry_function(), analysis=FAMILIES, level="cta", dims=dims
    )
    summary = get_metadata(result.function, PerformanceSummaryMetadata)
    assert summary is not None
    return summary.timeline.end_ns - summary.timeline.start_ns


@pytest.mark.parametrize("family", FAMILIES)
def test_the_result_names_the_function_that_carries_the_records(family: str) -> None:
    """The records are written onto the program measured, which is the derived one.

    Handing back the symbolic input would send a reader looking for records on a
    function that has none.
    """
    module, authored, dims = _subject(family)

    result = analyze(module, authored, analysis=family, dims=dims)

    assert result.function is not authored
    assert result.function.name == authored.name
    assert residual_dims(result.function) == ()


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
