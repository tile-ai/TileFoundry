"""The typed analysis families, exercised through the composed operation.

One semantic invariant per family, and per defect: what a family reports for a
real model is the corpus Analyze witness's subject, and whether it runs at all is
every other test's. What is left here is the arithmetic and the failure
semantics -- the work model from both sides, the two capacity verdicts that read
differently, the bound that must not be summed, and the report that must not
disagree with itself.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.analysis import (
    Breakdown,
    ComputeCostMetadata,
    MemoryHierarchyFacts,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceServiceFacts,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
    Spread,
    ThroughputFacts,
    Traffic,
)
from tilefoundry.analysis.api import _build_context, analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.compute_cost import (
    local_duration_ns,
)
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.memory import MemoryOptions
from tilefoundry.analysis.metadata import shifted
from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.inspection.analysis_report import render_analysis, render_text
from tilefoundry.ir.core import (
    Call,
    get_metadata,
)
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.contexts import TrafficBytes
from tilefoundry_nvidia import CudaTarget

PLACED = Path(__file__).parents[1] / "fixtures" / "placed"
ANALYSIS = Path(__file__).parents[1] / "fixtures" / "analysis"
ACCESS_FOOTPRINT = Path(__file__).parents[1] / "models" / "access_footprint" / "model.py"


class FamilyCase(Case):
    """One program whose family records state the arithmetic a family must keep."""


@case
class InvariantGemmOperands(FamilyCase):
    """The QKV tiles load again even when one operand is invariant in one loop."""

    path: Path = ACCESS_FOOTPRINT
    name: str | None = "TiledQKVProjection"

    def guard(self, state, problems):
        result = analyze(
            *check_program(state, state.entry_function()),
            analysis="memory",
        )
        record = get_metadata(result.function, RegionMemoryMetadata)
        gmem = record.traffic.storage.of("gmem")

        assert gmem is not None
        assert gmem.logical == TrafficBytes(read=27_262_976, write=16_777_216)
        assert gmem.total == TrafficBytes(read=18_824_036_352, write=2_214_592_512)


def test_reshard_aliases_input_buffer() -> None:
    TransposeShard = load_authored_ir(f"{PLACED / 'tp_all_to_all.py'}:TransposeShard")
    result = analyze(
        *check_program(TransposeShard, TransposeShard.entry_function()), analysis="memory"
    )
    (reshard,) = (
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, Reshard)
    )
    assert aliased_operand(reshard) == 0
    record = get_metadata(reshard, MemoryMetadata)
    assert record.buffer_bytes is None and not record.offsets
    assert record.traffic.communication.of("gpu") is None
    assert (
        get_metadata(result.function, RegionMemoryMetadata).peak_for("gmem").peak_bytes
        == load_namespace(str(PLACED / "tp_all_to_all.py"))[0]["HELD_BYTES"]
    )


_ALL_FAMILIES = ("compute-cost", "memory", "roofline", "performance")


class _RestatedCapacity(CudaTarget):
    """An H200 whose addressable levels hold what a test needs to say."""

    name = "test.restated_capacity"
    levels: dict[str, int] = {}

    def get_facts(self, facts_type: type, query: object | None = None):
        facts = super().get_facts(facts_type, query)
        if facts_type is not MemoryHierarchyFacts:
            return facts
        return replace(
            facts,
            explicit_levels=tuple(
                replace(level, capacity_bytes=self.levels.get(level.name, level.capacity_bytes))
                for level in facts.explicit_levels
            ),
        )


class _TightShared(_RestatedCapacity):
    name = "test.tight_shared"
    levels = {"smem": 105_600}


class _RoomyShared(_RestatedCapacity):
    name = "test.roomy_shared"
    levels = {"smem": 422_400}


class _RoomierShared(_RestatedCapacity):
    name = "test.roomier_shared"
    levels = {"smem": 845_000}


def _calls(function) -> tuple[Call, ...]:
    return tuple(expr for expr in collect_exprs(function.body) if isinstance(expr, Call))


@case
class CrossScopePerformance(FamilyCase):
    """The outer stage waits for the inner stage to complete."""

    path: Path = PLACED / "performance_findings.py"
    name: str | None = "_CrossScopePerformance"

    def guard(self, state, problems):
        results = {}
        for name in ("cross_scope", "nested_control"):
            function = next(fn for fn in state.functions if fn.name == name)
            results[name] = analyze(
                *check_program(state, function, dims={"seq_len": 8192}),
                analysis=_ALL_FAMILIES,
            )
            assert get_metadata(results[name].function, PerformanceSummaryMetadata) is not None

        cross_scope = results["cross_scope"]
        timed = []
        context = _build_context(cross_scope.module, cross_scope.function)[-1]
        for call in _calls(cross_scope.function):
            frame = context._callee_context(call)
            occurrence = get_metadata(call, PerformanceMetadata)
            if frame is None or occurrence is None:
                continue
            for primitive in _calls(frame.function):
                record = get_metadata(primitive, PerformanceMetadata)
                if record is None:
                    continue
                resolved = replace(
                    record, timeline=shifted(record.timeline, occurrence.timeline.start_ns)
                )
                for invocation in range(occurrence.timeline.trips):
                    for iteration in range(resolved.timeline.trips):
                        offset = (
                            invocation * occurrence.timeline.stride_ns
                            + iteration * resolved.timeline.stride_ns
                        )
                        timed.append(
                            (
                                primitive,
                                replace(
                                    resolved,
                                    timeline=replace(
                                        resolved.timeline,
                                        start_ns=resolved.timeline.start_ns + offset,
                                        end_ns=resolved.timeline.end_ns + offset,
                                    ),
                                ),
                            )
                        )
        scalar_indices = tuple(
            (call, record)
            for call, record in timed
            if record is not None and isinstance(call.target, Binary) and call.type.shape == ()
        )
        stores = tuple(
            (call, record)
            for call, record in timed
            if record is not None and isinstance(call.target, InsertSlice)
        )
        loads = tuple(
            (call, record)
            for call, record in timed
            if record is not None and isinstance(call.target, Reshard)
        )

        outer_index = max(scalar_indices, key=lambda item: item[1].timeline.start_ns)[1]
        inner_store = max(
            (
                record
                for _call, record in stores
                if record.timeline.end_ns <= outer_index.timeline.start_ns
            ),
            key=lambda record: record.timeline.end_ns,
        )
        outer_load = min(
            (
                record
                for _call, record in loads
                if record.timeline.start_ns >= outer_index.timeline.end_ns
            ),
            key=lambda record: record.timeline.start_ns,
        )

        assert inner_store.timeline.end_ns == outer_index.timeline.start_ns
        assert outer_index.timeline.end_ns == outer_load.timeline.start_ns


def test_symbolic_stride_adds_address_ops() -> None:
    """Group_index * 32 is an address, not an unbound parameter.

    Address arithmetic executes in each of the 32 position iterations.
    """
    observed = {}
    for name, root in (
        ("literal", "_LiteralStoreOffset"),
        ("symbolic", "_SymbolicStoreOffset"),
    ):
        owner = load_authored_ir(f"{PLACED / 'symbolic_offset.py'}:{root}")
        result = analyze(
            *check_program(owner, owner.entry_function(), dims={"seq_len": 8192}),
            analysis=_ALL_FAMILIES,
        )
        cost = get_metadata(result.function, ComputeCostMetadata)
        bound = get_metadata(result.function, RooflineMetadata)
        summary = get_metadata(result.function, PerformanceSummaryMetadata)
        assert cost is not None and bound is not None and summary is not None
        assert bound.memory_ns == 398_459
        integer_ops = cost.other_ops.of("integer")
        assert integer_ops is not None
        observed[name] = (
            integer_ops.total,
            integer_ops.per_unit[0],
            bound.ideal_ns,
            summary.timeline.end_ns - summary.timeline.start_ns,
        )

    literal_service, literal_local, literal_roofline, literal_performance = observed["literal"]
    symbolic_service, symbolic_local, symbolic_roofline, symbolic_performance = observed["symbolic"]
    assert literal_roofline == symbolic_roofline == 453_060
    assert symbolic_local - literal_local == 6 * 32
    assert symbolic_service - literal_service == 6 * 32 * 128
    assert symbolic_performance - literal_performance == 6 * 32


@case
class LargeRooflineRounding(FamilyCase):
    path: Path = ANALYSIS / "large_roofline_rounding.py"
    name: str | None = "_LargeRooflineRounding"

    def guard(self, state, problems):
        result = analyze(
            *check_program(state, state.entry_function()),
            analysis="roofline",
        )
        bound = get_metadata(result.function, RooflineMetadata)

        assert bound is not None
        assert bound.compute_ns == 197_014_925_373_451
        assert bound.ideal_ns == 197_014_925_373_451


_SPLIT_GRID = 128


def test_matmul_work_ignores_split_axis() -> None:
    """One gemm, two layouts of the weight, one per-unit answer.

    Split the result's last axis and the axis sharding adds lands where a batch
    is read from, so counting ``batch * m * n`` charges the rows twice and never
    divides by the grid. Counting the result's elements does not care where the
    axis went, and the two spellings agree on the work and the predicted time.
    """
    _SplitLastAxis = load_authored_ir(f"{ANALYSIS / 'split_gemm_last_axis.py'}:_SplitLastAxis")
    _SplitStripMajor = load_authored_ir(
        f"{ANALYSIS / 'split_gemm_strip_major.py'}:_SplitStripMajor"
    )
    per_layout = {}
    for owner, name in ((_SplitLastAxis, "last_axis"), (_SplitStripMajor, "strip_major")):
        function = next(item for item in owner.functions if item.name == name)
        report = analyze(*check_program(owner, function), analysis="performance")
        product = next(
            expr
            for expr in collect_exprs(report.function.body)
            if isinstance(expr, Call) and type(expr.target).__name__ == "MatMul"
        )
        cost = get_metadata(product, ComputeCostMetadata)
        summary = get_metadata(report.function, PerformanceSummaryMetadata)
        bf16 = cost.flops.of("bf16")
        assert bf16 is not None
        per_layout[name] = (
            bf16.total,
            bf16.per_unit[0],
            summary.timeline.end_ns,
        )

    for name, (whole, unit, _predicted) in per_layout.items():
        assert whole == unit * _SPLIT_GRID, f"{name} did not divide by the grid"
    assert per_layout["last_axis"] == per_layout["strip_major"]


def test_capacity_overflow_reports_error() -> None:
    """A placement error is report data rather than an aborted analysis.

    One shared tile of this program is twice what the tight machine states for
    that level. The solver still places it and its pointwise add result in one
    buffer, then reports that solved high-water against capacity. Restating
    capacity changes only the error, not the logical lifetimes or placement.
    """
    _SharedTile = load_authored_ir(f"{ANALYSIS / 'shared_tile.py'}:_SharedTile")
    tight = replace(_SharedTile, target=_TightShared("nvidia.h200_sxm"))
    split = next(function for function in tight.functions if function.name == "split")
    roomy = replace(_SharedTile, target=_RoomyShared("nvidia.h200_sxm"))

    tight_memory = analyze(*check_program(tight, split), analysis="memory")
    tight_record = get_metadata(tight_memory.function, RegionMemoryMetadata)
    assert tight_record.errors == ("smem placement peak 206.25KiB exceeds capacity 103.12KiB",)
    assert '#   error="smem placement peak 206.25KiB exceeds capacity 103.12KiB"' in render_text(
        render_analysis(tight_memory)
    )
    assert render_analysis(tight_memory).data["functions"][0]["function_records"]["memory"][
        "errors"
    ] == ["smem placement peak 206.25KiB exceeds capacity 103.12KiB"]

    tight_performance = analyze(*check_program(tight, split), analysis="performance")
    assert (
        get_metadata(tight_performance.function, RegionMemoryMetadata).errors == tight_record.errors
    )

    unrestated = next(item for item in _SharedTile.functions if item.name == "split")
    held = get_metadata(
        analyze(
            *check_program(_SharedTile, unrestated),
            analysis="memory",
            options=MemoryOptions(timeout_seconds=1.0),
        ).function,
        RegionMemoryMetadata,
    ).peaks
    assert next(item.peak_bytes for item in held if item.memory_level == "smem") == 211_200

    fits = analyze(
        *check_program(roomy, next((item for item in roomy.functions if item.name == "split"))),
        analysis="performance",
    )
    summary = get_metadata(fits.function, PerformanceSummaryMetadata)
    assert summary is not None
    fits_memory = get_metadata(fits.function, RegionMemoryMetadata)
    assert fits_memory.solver_status == "feasible"
    assert fits_memory.errors == ()
    assert summary.timeline.end_ns > 0

    wider = replace(_SharedTile, target=_RoomierShared("nvidia.h200_sxm"))
    relieved = analyze(
        *check_program(wider, next((item for item in wider.functions if item.name == "split"))),
        analysis="memory",
    )
    assert [
        (type(item.value.expr), item.value.index, item.memory_level, item.bytes, item.defined_at, item.last_used_at)
        for item in get_metadata(fits.function, RegionMemoryMetadata).lifetimes
    ] == [
        (type(item.value.expr), item.value.index, item.memory_level, item.bytes, item.defined_at, item.last_used_at)
        for item in get_metadata(relieved.function, RegionMemoryMetadata).lifetimes
    ]


def test_local_duration_prices_stated_rates() -> None:
    """Compute holes fail while unstated memory levels remain unpriced.

    A compute kind promises work the machine must price, so a missing rate is an
    error. Memory traffic is timed only at levels with a stated bandwidth; the
    independently overlappable levels set a maximum rather than a sum.
    """
    _PricingBoundary = load_authored_ir(f"{ANALYSIS / 'pricing_boundary.py'}:_PricingBoundary")
    target = _PricingBoundary.resolve_target()
    throughput = target.get_facts(ThroughputFacts)
    services = target.get_facts(PerformanceServiceFacts)
    result = analyze(
        *check_program(_PricingBoundary, _PricingBoundary.functions[0]),
        analysis=("compute-cost", "memory"),
    )
    records = [get_metadata(call, ComputeCostMetadata) for call in _calls(result.function)]
    work = next(
        record
        for record in records
        if record is not None and any(spread.per_unit[0] for _name, spread in record.flops.kinds)
    )

    with pytest.raises(
        AnalysisError,
        match=r"^performance: selected topology level 'thread', but the target's "
        r"one-unit throughputs are stated for 'cta'$",
    ):
        local_duration_ns(work, throughput, services, level="thread")

    with pytest.raises(AnalysisError, match=r"unknown compute dtype 'f9e9m9'"):
        local_duration_ns(
            replace(
                work,
                flops=Breakdown(
                    (
                        *work.flops.kinds,
                        ("f9e9m9", Spread(total=8, per_unit=(8,), logical_groups=(((), 8),))),
                    )
                ),
            ),
            throughput,
            services,
            level="cta",
        )

    crossed = MemoryMetadata(
        topologies=("cta",),
        traffic=Traffic(
            storage=Breakdown(
                (
                    (
                        throughput.bandwidth_level,
                        Spread(
                            total=TrafficBytes(read=4096),
                            per_unit=(TrafficBytes(read=4096),),
                            logical_groups=(((), TrafficBytes(read=4096)),),
                        ),
                    ),
                    (
                        "smem",
                        Spread(
                            total=TrafficBytes(read=8192),
                            per_unit=(TrafficBytes(read=8192),),
                            logical_groups=(((), TrafficBytes(read=8192)),),
                        ),
                    ),
                )
            ),
        ),
    )
    assert (
        local_duration_ns(
            ComputeCostMetadata(),
            throughput,
            replace(services, unit_bandwidth=()),
            moved=crossed,
            level="cta",
        )
        == 0
    )
    priced = replace(
        services,
        unit_bandwidth=((throughput.bandwidth_level, 1024_000_000_000), ("smem", 1024_000_000_000)),
    )
    assert (
        local_duration_ns(ComputeCostMetadata(), throughput, priced, moved=crossed, level="cta")
        == 8
    )


CASES = special(FamilyCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_family_case(case: FamilyCase) -> None:
    check_case(case)
