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
from tilefoundry.analysis.walk import collect_exprs
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
    return tuple(expr for expr in collect_exprs(function.body) if isinstance(expr, Call))


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


_SPLIT_GRID = 128
_SPLIT_HIDDEN = 2048
_SPLIT_OUT = 12288
_SPLIT_PER = _SPLIT_OUT // _SPLIT_GRID
_SPLIT_BLOCK = 128


@module(entry="last_axis", target=_H200, topologies=(Topology("cta", _SPLIT_GRID),))
class _SplitLastAxis:
    """The weight's N on the mesh, so the result's last axis is the split one."""

    @func
    def last_axis(
        x: Tensor[(1, _SPLIT_BLOCK, _SPLIT_HIDDEN), "bf16"],
        w: ConstTensor[(_SPLIT_HIDDEN, _SPLIT_OUT), "bf16"],
    ) -> Tensor[(1, _SPLIT_BLOCK, _SPLIT_OUT), "bf16"]:
        with Mesh(("cta",), layout=(_SPLIT_GRID,), names=("unit",)) as mesh:
            rows = tf.reshard(
                x[:, :, 0:_SPLIT_BLOCK], (1, _SPLIT_BLOCK, _SPLIT_BLOCK), "smem"
            )
            strip = tf.reshard(
                w[0:_SPLIT_BLOCK, :], (_SPLIT_BLOCK, _SPLIT_OUT @ mesh.unit), "smem"
            )
            return tf.matmul(rows, strip)


@module(entry="strip_major", target=_H200, topologies=(Topology("cta", _SPLIT_GRID),))
class _SplitStripMajor:
    """The same gemm with the split axis leading, inside the matmul's batch."""

    @func
    def strip_major(
        x: Tensor[(1, _SPLIT_BLOCK, _SPLIT_HIDDEN), "bf16"],
        w: ConstTensor[(_SPLIT_GRID, _SPLIT_HIDDEN, _SPLIT_PER), "bf16"],
    ) -> Tensor[(_SPLIT_GRID, _SPLIT_BLOCK, _SPLIT_PER), "bf16"]:
        with Mesh(("cta",), layout=(_SPLIT_GRID,), names=("unit",)) as mesh:
            rows = tf.reshard(
                x[:, :, 0:_SPLIT_BLOCK], (1, _SPLIT_BLOCK, _SPLIT_BLOCK), "smem"
            )
            strip = tf.reshard(
                w[:, 0:_SPLIT_BLOCK, :],
                (_SPLIT_GRID @ mesh.unit, _SPLIT_BLOCK, _SPLIT_PER),
                "smem",
            )
            return tf.matmul(rows, strip)


def test_a_matmul_counts_its_rows_once_whichever_axis_the_mesh_split() -> None:
    """One gemm, two layouts of the weight, one per-unit answer.

    Split the result's last axis and the axis sharding adds lands where a batch
    is read from, so counting ``batch * m * n`` charges the rows twice and never
    divides by the grid. Counting the result's elements does not care where the
    axis went, and the two spellings agree on the work and the predicted time.
    """
    per_layout = {}
    for owner, name in ((_SplitLastAxis, "last_axis"), (_SplitStripMajor, "strip_major")):
        function = next(item for item in owner.functions if item.name == name)
        report = analyze(owner, function, analysis="performance")
        product = next(
            expr
            for expr in collect_exprs(report.function.body)
            if isinstance(expr, Call) and type(expr.target).__name__ == "MatMul"
        )
        cost = get_metadata(product, ComputeCostMetadata)
        summary = get_metadata(report.function, PerformanceSummaryMetadata)
        per_layout[name] = (
            dict(cost.flops)["bf16"],
            dict(cost.flops_per_unit)["bf16"],
            summary.timeline.end_ns,
        )

    for name, (whole, unit, _predicted) in per_layout.items():
        assert whole == unit * _SPLIT_GRID, f"{name} did not divide by the grid"
    assert per_layout["last_axis"] == per_layout["strip_major"]


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
