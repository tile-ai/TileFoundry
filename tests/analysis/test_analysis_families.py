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

import pytest

from tests.models.corpus import placed_cases, placed_fixture_roots
from tilefoundry import func, module
from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryHierarchyFacts,
    MemoryMetadata,
    PerformanceServiceFacts,
    PerformanceSummaryMetadata,
    RooflineMetadata,
    ThroughputFacts,
    TrafficMetadata,
)
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.compute_cost import (
    _local_duration_ns,
)
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.walk import postorder
from tilefoundry.dsl import ConstTensor, DimVar, Mesh, Tensor, Topology, tf
from tilefoundry.ir.core import (
    Call,
    get_metadata,
)
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.contexts import TrafficBytes

_ROUNDING_M = 14_593
_ROUNDING_N = 11_489
_ROUNDING_K = 298_224_413
_H200 = CudaTarget("nvidia.h200_sxm")
_LARGE_H200 = CudaTarget(
    replace(_H200.device, hbm_capacity_bytes=300_000_000_000_000_000),
    architecture=_H200.architecture,
)


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


class _RoomyShared(_RestatedCapacity):
    name = "test.roomy_shared"
    levels = {"smem": 422_400}


class _RoomierShared(_RestatedCapacity):
    name = "test.roomier_shared"
    levels = {"smem": 845_000}


_GRID = DimVar("grid", 1, 397)


@module(entry="main", target=_LARGE_H200, topologies=(Topology("cta", 1),))
class _LargeRooflineRounding:
    @func
    def main(
        lhs: Tensor[(_ROUNDING_M, _ROUNDING_K), "f32"],
        rhs: ConstTensor[(_ROUNDING_K, _ROUNDING_N), "f32"],
    ):
        return tf.matmul(lhs, rhs)


@module(
    entry="unpriced_only",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class _PricingBoundary:
    @func
    def unpriced_only(source: Tensor[(1,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
            local = tf.reshard(source, (1 @ cta.tile,), "rmem")
            moved = tf.reshard(local, (1 @ cta.tile,), "rmem")
            return tf.add(moved, moved)

    @func
    def mixed(source: Tensor[(64,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
            placed = tf.reshard(source, (64 @ cta.tile,), "gmem")
            local = tf.reshard(placed, (64 @ cta.tile,), "rmem")
            return tf.add(local, local)

    @func
    def serviced_predicate(source: Tensor[(64,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
            placed = tf.reshard(source, (64 @ cta.tile,), "gmem")
            return placed <= placed


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

    @func
    def viewed(source: Tensor[(1056, 6600), "f32"]):
        with Mesh(("cta",), layout=(132,), names=("tile",)) as cta:
            local = tf.reshard(source, (1056 @ cta.tile, 6600), "smem")
            view = tf.reshape(local, new_shape=(1056, 100, 66))
            return tf.add(view, view)


def _calls(function) -> tuple[Call, ...]:
    return tuple(expr for expr in postorder(function.body) if isinstance(expr, Call))


def test_roofline_uses_exact_integer_ceiling_above_float_precision() -> None:
    result = analyze(
        _LargeRooflineRounding,
        _LargeRooflineRounding.entry_function(),
        analysis="roofline",
    )
    bound = get_metadata(result.function, RooflineMetadata)

    assert bound is not None
    assert bound.compute_ns == 1_492_537_313_434
    assert bound.ideal_ns == 1_492_537_313_434


def test_a_program_whose_buffers_have_nowhere_to_sit_is_refused() -> None:
    """Placing the buffers is what makes the rest of the answer worth having.

    The two shared tiles this program keeps live at once do not both fit the
    machine's shared memory, and no ordering makes them: the second is computed
    from the first. Whoever asks is refused, because memory is the family that
    decides this and everything downstream reads what it decided. Restating the
    capacity is what changes the answer, and it changes only the answer: the
    lifetimes are the same either way.
    """
    split = next(function for function in _SharedTile.functions if function.name == "split")
    roomy = replace(_SharedTile, target=_RoomyShared("nvidia.h200_sxm"))

    with pytest.raises(AnalysisError, match=r"'smem' holds 422400 B at one point"):
        analyze(_SharedTile, split, analysis="memory")
    with pytest.raises(AnalysisError, match=r"'smem' holds 422400 B at one point"):
        analyze(_SharedTile, split, analysis="performance")

    fits = analyze(
        roomy,
        next(item for item in roomy.functions if item.name == "split"),
        analysis="performance",
    )
    summary = get_metadata(fits.function, PerformanceSummaryMetadata)
    assert summary is not None
    assert get_metadata(fits.function, MemoryMetadata).allocation.solver_status == "optimal"
    assert summary.timeline.end_ns > 0

    wider = replace(_SharedTile, target=_RoomierShared("nvidia.h200_sxm"))
    relieved = analyze(
        wider,
        next(item for item in wider.functions if item.name == "split"),
        analysis="memory",
    )
    assert [
        (item.binding, item.level, item.bytes, item.defined_at, item.last_used_at)
        for item in get_metadata(fits.function, MemoryMetadata).lifetimes
    ] == [
        (item.binding, item.level, item.bytes, item.defined_at, item.last_used_at)
        for item in get_metadata(relieved.function, MemoryMetadata).lifetimes
    ]


def test_the_placed_inventory_takes_the_whole_directory() -> None:
    """What is asked about is what is there, and what is left out is named.

    The inventory is walked out of the fixture package rather than listed, so a
    fixture added to it joins instead of quietly escaping. This pins the walk --
    every root found, a Module reached as somebody's child not counted as one,
    and the only roots left out being those whose machine runs no CTAs. Excluding
    one for any other reason is the allowlist this prevents, and would show up
    here as a number that moved. What those cases answer is asked where the
    analyses run; a count is not an answer, so this runs nothing.
    """
    roots = placed_fixture_roots()
    assert len(roots) == 29

    outside = []
    for file, name, published in roots:
        root = published
        try:
            root.resolve_target()
        except Exception:  # noqa: BLE001 -- a root that names no machine
            root = replace(root, target=CudaTarget("nvidia.h200_sxm"))
        if "cta" not in root.resolve_target().topology_levels:
            outside.append(f"{file}.{name}")
    assert outside == [
        "fused_twin.Fused",
        "nested_twin.Nested",
        "square_cpu.Mine",
        "square_cpu_runtime.Mine",
        "square_twin.Model",
        "weighted_twin.WeightedRoot",
    ]

    cases = placed_cases()
    assert len(roots) - len(outside) == 23, "the roots that answer for CTAs"
    assert len({case.id.rsplit("[", 1)[0] for case in cases}) == 29, (
        "one row per selector those roots expose, not one per root"
    )
    assert len(cases) == 32, "and one case per stated set of sizes"


def test_a_price_is_refused_where_the_machine_states_no_rate_to_pay_it_at() -> None:
    """A hole inside a number reads as a program that does less than it does.

    Three ways the target can fail to price what a program asks for, and none of
    them may be answered with nothing: rates stated for the wrong unit, a dtype
    with no throughput, and the one level a bandwidth was meant for having none.
    Each says which quantity and which level, because that is what a reader has
    to go and publish.
    """
    target = _PricingBoundary.resolve_target()
    throughput = target.get_facts(ThroughputFacts)
    services = target.get_facts(PerformanceServiceFacts)
    result = analyze(
        _PricingBoundary, _PricingBoundary.functions[0], analysis=("compute-cost", "memory")
    )
    records = [get_metadata(call, ComputeCostMetadata) for call in _calls(result.function)]
    work = next(
        record
        for record in records
        if record is not None and any(v for _n, v in record.flops_per_unit)
    )

    with pytest.raises(
        AnalysisError,
        match=r"^performance: selected topology level 'thread', but the target's "
        r"one-unit throughputs are stated for 'cta'$",
    ):
        _local_duration_ns(work, throughput, services, level="thread")

    with pytest.raises(AnalysisError, match=r"unknown compute dtype 'f9e9m9'"):
        _local_duration_ns(
            replace(work, flops_per_unit=(("f9e9m9", 8),)),
            throughput,
            services,
            level="cta",
        )

    crossed = TrafficMetadata(
        per_unit=((throughput.bandwidth_level, TrafficBytes(read=4096)),)
    )
    with pytest.raises(
        AnalysisError,
        match=rf"no one-unit throughput for level '{throughput.bandwidth_level}' at 'cta'",
    ):
        _local_duration_ns(
            ComputeCostMetadata(),
            throughput,
            replace(services, unit_bandwidth=()),
            moved=crossed,
            level="cta",
        )
