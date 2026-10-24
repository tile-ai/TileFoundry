"""Shared report, internal-record, and persistent-schedule assertions."""

from __future__ import annotations

from dataclasses import dataclass

import isl

from tilefoundry.analysis import (
    AnalysisPrecision,
    AnalysisResult,
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceMetadata,
)
from tilefoundry.analysis.access import Access, eliminate_parameters
from tilefoundry.analysis.compute_cost import inflight_bytes, local_duration_ns
from tilefoundry.analysis.iteration_scope import IterationScope
from tilefoundry.ir.core import Call, describe_expr, get_metadata
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import PerformanceServiceFacts, ThroughputFacts, TopologyFacts
from tilefoundry.visitor_registry.contexts import ValuationContext


@dataclass(frozen=True)
class PersistentSchedule:
    loop_trips: tuple[tuple[str, int], ...]
    store_loop: str
    store_precision: AnalysisPrecision
    compared_units: tuple[tuple[int, ...], tuple[int, ...]]


def _loop_scopes(scopes: tuple[IterationScope, ...]) -> dict[str, IterationScope]:
    return {
        scope.owner.induction_var.name: scope
        for scope in scopes
        if isinstance(scope.owner, LoopRegion)
    }


def _insert_slice_output(scope: IterationScope) -> Access:
    for call, accesses in scope.access.outputs.get("narrow", {}).values():
        if isinstance(call.target, InsertSlice):
            assert len(accesses) == 1
            return accesses[0]
    raise AssertionError("loop has no InsertSlice output")


def _at_unit(image: isl.set, coordinates: tuple[int, ...]) -> isl.set:
    assert image.dim(isl.dim_type.PARAM) == len(coordinates)
    for axis, coordinate in enumerate(coordinates):
        image = image.fix_si(isl.dim_type.PARAM, axis, coordinate)
    return image


def _assert_persistent_schedule(
    scopes: tuple[IterationScope, ...],
    expected: PersistentSchedule,
) -> None:
    loops = _loop_scopes(scopes)
    for name, trips in expected.loop_trips:
        assert loops[name].trips() == trips

    store = _insert_slice_output(loops[expected.store_loop])
    scope = loops[expected.store_loop]
    relation, precision = eliminate_parameters(
        store.relation, scope, store.target.leaf(), narrow=True
    )
    assert precision is expected.store_precision
    written = relation.project(
        tuple(loop.owner.induction_var for loop in scope.loop_scopes)
    ).relation.range()
    first, second = (_at_unit(written, unit) for unit in expected.compared_units)
    assert first.is_disjoint(second)


def assert_reported_contract(report: dict) -> None:
    """Every performance conclusion traces back to what it was derived from.

    The prediction contains each occurrence it timed and is no faster than the
    ideal bound, and a solve that proved nothing says so. One a loop repeats is
    written once, and its last trip still lands inside the prediction that
    contains it. A loop is not an occurrence and carries no timeline of its own.
    """
    records = report["functions"][0]["function_records"]
    summary = records["performance"]
    timeline = summary["timeline"]
    assert 0 <= timeline["start_ns"] <= timeline["end_ns"]
    assert records["memory"]["solver_status"] in ("optimal", "feasible")
    predicted_ns = timeline["end_ns"] - timeline["start_ns"]
    assert predicted_ns % -(-summary["wave"][0] // summary["wave"][1]) == 0
    assert records["roofline"]["ideal_ns"] <= predicted_ns

    timed = 0
    for call in report["functions"][0]["calls"]:
        record = call.get("performance")
        if record is None:
            continue
        timed += 1
        occurrence = record["timeline"]
        start, end = occurrence["start_ns"], occurrence["end_ns"]
        assert timeline["start_ns"] <= start <= end <= timeline["end_ns"], call["value"]
        trips, stride = occurrence["trips"], occurrence["stride_ns"]
        assert trips >= 1, call["value"]
        assert (stride == 0) if trips == 1 else (stride >= end - start), call["value"]
        assert end + (trips - 1) * stride <= timeline["end_ns"], call["value"]
    assert bool(timed) is bool(predicted_ns)
    for loop in report["functions"][0]["loops"]:
        assert "performance" not in loop, loop["value"]

    def nonnegative(value: object) -> None:
        """Every quantity these four families report is a count, so none is below zero.

        Work, moved bytes, placement peaks and a bound are all counts of something that
        happened or has to happen. A negative one is not a small answer but a
        derivation that ran backwards -- a projection dividing what it should have
        multiplied, or a difference taken the wrong way round -- and it would then be
        added into a total that still looks plausible.
        """
        if isinstance(value, dict):
            for item in value.values():
                nonnegative(item)
        elif isinstance(value, list):
            for item in value:
                nonnegative(item)
        elif isinstance(value, (int, float)):
            assert value >= 0, value

    nonnegative(report)
    for lifetime in records["memory"]["lifetimes"]:
        assert 0 <= lifetime["defined_at"] <= lifetime["last_used_at"]
        assert "<buffer " not in lifetime["binding"]


def assert_internal_contract(result: AnalysisResult, scopes: tuple[IterationScope, ...]) -> None:
    """Each duration matches its priced work and divides the enclosing loop trips.

    A definition-local duration is its own compute-cost record priced at the
    target's rates for the units its definition runs. One a loop repeats is
    written once, so its interval is that many of its own durations.
    """
    module_target = result.module.resolve_target()
    throughput = module_target.get_facts(ThroughputFacts)
    physical = module_target.get_facts(TopologyFacts).level(result.level).max_physical_units
    for function, expr in (
        (function, expr) for function in result.functions for expr in collect_exprs(function.body)
    ):
        if (
            not isinstance(expr, Call)
            or ValuationContext(result.module, function)._callee_context(expr) is not None
        ):
            continue
        cost = get_metadata(expr, ComputeCostMetadata)
        assert cost is not None
        owner = next(
            scope.snapshot.definition.locations[id(expr)]
            for scope in scopes
            if id(expr) in scope.snapshot.definition.locations
        )
        active = min(owner.root.active_positions(result.level), physical)
        inflight = inflight_bytes(expr, owner, module_target, throughput.bandwidth_level)
        duration = local_duration_ns(
            cost,
            throughput,
            module_target.get_facts(PerformanceServiceFacts, (result.level, active, inflight)),
            moved=get_metadata(expr, MemoryMetadata),
            level=result.level,
            positions=min(owner.mesh_positions(result.level), active),
        )
        record = get_metadata(expr, PerformanceMetadata)
        if not duration:
            assert record is None
            continue
        assert record is not None
        span = record.timeline.end_ns - record.timeline.start_ns
        assert span % duration == 0, describe_expr(expr)
        runs = span // duration
        available = 1
        cursor = owner
        while cursor.parent is not None:
            available *= cursor.trips()
            cursor = cursor.parent
        assert 1 <= runs <= available and available % runs == 0, describe_expr(expr)
        trips = record.timeline.trips
        assert 1 <= trips <= available and available % trips == 0, describe_expr(expr)


def _footprint_bytes(memory: dict, name: str) -> int:
    return memory["footprint"]["buffers"][name]["gmem"]["logical"]


def _working_set_bytes(memory: dict) -> int:
    return sum(
        level["logical"]
        for levels in memory["footprint"]["buffers"].values()
        for level in levels.values()
    )


def _reuse_conclusions(memory: dict) -> list[dict]:
    fields = ("buffer", "time", "space", "holds_bytes", "reuse_bytes", "fits")
    return [{field: row[field] for field in fields} for row in memory["reuse_windows"]]
