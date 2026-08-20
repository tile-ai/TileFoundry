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
from dataclasses import replace

import pytest

import tilefoundry.analysis.compute_cost as compute_cost
from tests.fixtures.placed import performance_findings as findings
from tests.fixtures.placed.derived_prefill import DerivedPrefill
from tests.fixtures.placed.flash_split_k_decode import FlashSplitKDecode
from tests.fixtures.placed.moe_mega_kernel import MoEMegaKernel
from tests.fixtures.placed.prefill_decode_attention import PrefillDecodeAttention
from tests.fixtures.placed.square_cuda import Model as SquareCuda
from tests.fixtures.shapes.window_programs import WindowCost
from tests.models.corpus import placed_cases, placed_fixture_roots
from tests.models.registry import CORPUS
from tilefoundry import func, module
from tilefoundry.analysis import (
    AllocationMetadata,
    ComputeCostMetadata,
    LoopFootprintMetadata,
    MemoryHierarchyFacts,
    MemoryMetadata,
    MemoryRelationKind,
    ParallelCapacityFacts,
    PerformanceMetadata,
    PerformanceServiceFacts,
    PerformanceSummaryMetadata,
    RooflineMetadata,
    ThroughputFacts,
    TimelineMetadata,
    TrafficMetadata,
)
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import (
    _call_placements,
    _execution_placement,
    _mesh_image,
    _result_placement,
)
from tilefoundry.analysis.compute_cost import (
    _is_structural_occurrence,
    _local_duration_ns,
)
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.movement import (
    _call_movement,
)
from tilefoundry.analysis.roofline import _cost_bound
from tilefoundry.analysis.walk import postorder
from tilefoundry.dsl import ConstTensor, DimVar, Mesh, Tensor, Topology, tf
from tilefoundry.inspection.analysis_report import (
    render_analysis,
    render_json,
    render_text,
    report,
)
from tilefoundry.inspection.values import render_comment
from tilefoundry.ir.core import (
    Call,
    Constant,
    ExecutionDomainMetadata,
    Tuple,
    Var,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.hir import Function, GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.types import DType, TensorType, TupleType, make_tensor_type, numel, tensor_bytes
from tilefoundry.ir.types.shard import (
    Broadcast,
    ComposedLayout,
    Layout,
    ShardLayout,
)
from tilefoundry.ir.types.shard import (
    Mesh as IrMesh,
)
from tilefoundry.ir.types.shard import (
    Topology as IrTopology,
)
from tilefoundry.ir.types.shard.layout_algebra import size
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import AmxTarget, CudaTarget
from tilefoundry.visitor_registry import cost_evaluator_registry
from tilefoundry.visitor_registry.contexts import Cost, CostContext, TrafficBytes
from tilefoundry.visitor_registry.visitors import CostEvaluator


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),))
class _CudaAdd:
    @func
    def main(source: Tensor[(256,), "f32"]):
        return tf.add(source, source)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("thread", 32),))
class _NoParallelLevel:
    """A program that never names the level its machine runs work at."""

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


_ROUNDING_M = 14_593
_ROUNDING_N = 11_489
_ROUNDING_K = 298_224_413
_H200 = CudaTarget("nvidia.h200_sxm")
_LARGE_H200 = CudaTarget(
    replace(_H200.device, hbm_capacity_bytes=300_000_000_000_000_000),
    architecture=_H200.architecture,
)


class _ConstrainedCudaTarget(CudaTarget):
    name = "test.constrained_cuda"

    def get_facts(self, facts_type: type, query: object | None = None):
        facts = super().get_facts(facts_type, query)
        if facts_type is ParallelCapacityFacts:
            return replace(facts, parallel_units=64)
        return facts


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


class _NarrowRegisters(_RestatedCapacity):
    name = "test.narrow_registers"
    levels = {"rmem": 4}


_GRID = DimVar("grid", 1, 397)


@module(
    entry="main",
    target=_H200,
    topologies=(Topology("cta", _GRID),),
)
class _ScalableGrid:
    @func
    def main(source: Tensor[(64,), "f32"]):
        with Mesh(("cta",), layout=(_GRID,), names=("tile",)) as _cta:
            local = tf.reshard(source, (64,), "gmem")
            return tf.relu(local)


@module(entry="main", target=_LARGE_H200, topologies=(Topology("cta", 1),))
class _LargeRooflineRounding:
    @func
    def main(
        lhs: Tensor[(_ROUNDING_M, _ROUNDING_K), "f32"],
        rhs: ConstTensor[(_ROUNDING_K, _ROUNDING_N), "f32"],
    ):
        return tf.matmul(lhs, rhs)


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
        return tf.add(source, tf.zeros(Tensor[(64,), "f32"]))


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

    @func
    def tiled(source: Tensor[(1024, 2048), "f32"]):
        return tf.concat(source[0:512, :], source[512:1024, :], axis=0)

    @func
    def swapped(source: Tensor[(1024, 2048), "f32"]):
        return tf.concat(source[512:1024, :], source[0:512, :], axis=0)

    @func
    def held(source: Tensor[(1024, 2048), "f32"]):
        made = tf.add(source, source)
        window = made[0:256, :]
        return tf.add(window, window)

    @func
    def overwritten(source: Tensor[(1024, 2048), "f32"], patch: Tensor[(256, 2048), "f32"]):
        made = tf.add(source, source)
        return tf.insert_slice(made, patch, (0, 0))

    @func
    def read_after_write(
        source: Tensor[(1024, 2048), "f32"], patch: Tensor[(256, 2048), "f32"]
    ):
        made = tf.add(source, source)
        written = tf.insert_slice(made, patch, (0, 0))
        return tf.add(written, made)

    @func
    def written_through_a_view(
        source: Tensor[(1024, 2048), "f32"], patch: Tensor[(256, 2048), "f32"]
    ):
        with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
            made = tf.add(source, source)
            viewed = tf.reshard(made, (1024 @ cta.tile, 2048), "gmem")
            written = tf.insert_slice(viewed, patch, (0, 0))
            return tf.add(written, tf.reshard(made, (1024 @ cta.tile, 2048), "gmem"))

    @func
    def donated(source: Tensor[(1024, 2048), "f32"], patch: Tensor[(256, 2048), "f32"]):
        return tf.insert_slice(source, patch, (0, 0))


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _UnmaterializedValueCosts:
    @func
    def main(source: Tensor[(8, 4), "f32"]):
        shape_value = tf.shape_of(source)[1]
        return tf.zeros(Tensor[(shape_value,), "i64", "umat"])


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _RuntimeCoordinateCosts:
    @func
    def main(source: Tensor[(8, 4), "f32"], start: Tensor[(), "i64"]):
        return source[start : start + 4, :]


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _UnmaterializedIndexTensorCosts:
    @func
    def main():
        row_positions = tf.reshape(tf.arange(Tensor[(128,), "i64", "gmem"]), new_shape=(1, 128))
        column_positions = tf.reshape(tf.arange(Tensor[(128,), "i64", "gmem"]), new_shape=(128, 1))
        return row_positions <= column_positions


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _IndexSelected:
    @func
    def main(
        table: ConstTensor[(1024, 64), "f32"],
        rows: ConstTensor[(4,), "i32"],
    ):
        return tf.index_select(table, rows, dim=0)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),))
class _OverwrittenAcrossGroups:
    """A view of an overwritten buffer, read where the overwrite does not run.

    The reader sits on the CTAs the write does not touch, so nothing about
    participants keeps the two apart. What has to keep them apart is that they
    are the same bytes: the reader reads a reshard of the buffer the write
    replaces.
    """

    @func
    def main(source: Tensor[(64,), "f32"], patch: Tensor[(32,), "f32"]):
        with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
            with cta[2:] as high:
                placed = tf.reshard(source, (64 @ high.tile,), "gmem")
                made = tf.add(placed, placed)
            with cta[:2] as low:
                moved = tf.reshard(made, (64 @ low.tile,), "gmem")  # noqa: F821
                side = tf.add(moved, moved)
            with cta[2:] as again:
                window = tf.reshard(patch, (32 @ again.tile,), "gmem")
                written = tf.insert_slice(made, window, (0,))  # noqa: F821
            return tf.add(
                tf.reshard(side, (64,), "gmem"),  # noqa: F821
                tf.reshard(written, (64,), "gmem"),  # noqa: F821
            )


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


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 168),))
def _placed_wide_grid(source: Tensor[(1024,), "f32"]):
    with Mesh(("cta",), layout=(168,), names=("tile",)) as _cta:
        placed = tf.reshard(source, (1024,), "gmem")
        return tf.add(placed, placed)


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


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def _zero_cost_view(source: Tensor[(1,), "f32"]):
    with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
        local = tf.reshard(source, (1 @ cta.tile,), "gmem")
        parts = tf.split(local, axis=0, num_splits=1)
        return parts[0]


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def _no_op_reshard(source: Tensor[(1,), "f32"]):
    return tf.reshard(source)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 132),))
def _replicated_source(x: Tensor[(16896,), "f32"], bias: Tensor[(1,), "f32"]):
    with Mesh(("cta",), layout=(132,), names=("block",)) as m:
        local = tf.reshard(x, (16896 @ m.block,), "rmem")
        spread = tf.reshard(bias, (1,), "rmem")
        return tf.reshard(tf.add(local, spread), (16896 @ m.block,), "gmem")


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 132),))
def _replicated_tuple(x: Tensor[(132, 128), "f32"]):
    with Mesh(("cta",), layout=(132,), names=("block",)) as m:
        local = tf.reshard(x, (132 @ m.block, 128), "rmem")
        chosen, _found = tf.topk(local, k=4, axis=-1)
        return tf.reshard(chosen, (132 @ m.block, 4), "gmem")


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def _unplaced_structural(source: Tensor[(), "i64", None, "rmem"]):
    return tf.stack(source, axis=0)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 2), Topology("thread", 4)))
def _thread_sharded(source: Tensor[(8,), "f32"]):
    with Mesh(("thread",), layout=(4,), names=("lane",)) as thread:
        local = tf.reshard(source, (8 @ thread.lane,), "rmem")
        return tf.add(local, local)


@func(
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 2)),
)
def _multi_topology_mesh(source: Tensor[(4,), "f32"]):
    with Mesh(("cta", "thread"), layout=(4,), names=("position",)) as mesh:
        local = tf.reshard(source, (4 @ mesh.position,), "rmem")
        return tf.add(local, local)


@func(
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 2)),
)
def _segmented_mesh(source: Tensor[(8,), "f32"]):
    with Mesh(("cta", "thread"), layout=(2, 2), names=("block", "lane")) as mesh:
        local = tf.reshard(source, (8 @ (mesh.block, mesh.lane),), "rmem")
        return tf.reshard(tf.add(local, local), (8 @ (mesh.block, mesh.lane),), "gmem")


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


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("thread", 4),))
def _modest_shared(source: Tensor[(1024,), "f32"]):
    with Mesh(("thread",), layout=(4,), names=("lane",)) as thread:
        local = tf.reshard(source, (1024 @ thread.lane,), "smem")
        return tf.add(local, local)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),))
def _oversized_working_set(source: Tensor[(16_000_000,), "f32"]):
    with Mesh(("cta",), layout=(4,), names=("tile",)) as _cta:
        broadcast = tf.reshard(source, (16_000_000,), "gmem")
        result = broadcast
        for i in tile(2, 1):  # noqa: F405
            result = tf.add(result, broadcast)
        return result


def _entry(owner):
    """The Function a Module is entered through."""
    return owner.entry_function()


def _run(owner, analysis: str):
    """Analyse *owner*'s entry function and hand back the record-bearing view."""
    entry = _entry(owner)
    result = analyze(owner, entry, analysis=analysis)
    return result, result.function


def _calls(function) -> tuple[Call, ...]:
    return tuple(expr for expr in postorder(function.body) if isinstance(expr, Call))


def _cost_of(function) -> ComputeCostMetadata:
    (record,) = [get_metadata(call, ComputeCostMetadata) for call in _calls(function)[-1:]]
    assert record is not None
    return record


def test_compute_cost_stops_at_the_selected_topology_level() -> None:
    entry = _entry(_thread_sharded)
    cta_result = analyze(_thread_sharded, entry, analysis=("compute-cost", "memory"))

    record = get_metadata(_calls(cta_result.function)[-1], ComputeCostMetadata)
    record_moved = get_metadata(_calls(cta_result.function)[-1], TrafficMetadata)
    assert record is not None
    assert cta_result.level == "cta"
    assert record.flops == (("f32", 8),)
    assert record.flops_per_unit == (("f32", 8),)
    traffic = record_moved.whole

    thread_result = analyze(_thread_sharded, entry, analysis=("compute-cost", "memory"), level="thread")
    record = get_metadata(_calls(thread_result.function)[-1], ComputeCostMetadata)
    record_moved = get_metadata(_calls(thread_result.function)[-1], TrafficMetadata)
    assert record is not None
    assert thread_result.level == "thread"
    assert record.flops == (("f32", 8),)
    assert record.flops_per_unit == (("f32", 2),)
    assert record_moved.whole == traffic


def test_an_unsharded_call_reports_the_same_global_and_per_unit_work() -> None:
    cuda_entry = _CudaAdd.entry_function()
    amx_entry = _AmxAdd.entry_function()

    cuda = analyze(_CudaAdd, cuda_entry, analysis=("compute-cost", "memory"))
    amx = analyze(_AmxAdd, amx_entry, analysis=("compute-cost", "memory"))

    call = _calls(cuda.function)[-1]
    record = get_metadata(call, ComputeCostMetadata)
    assert record is not None
    assert record.flops == (("f32", numel(call.type)),)
    assert record.flops_per_unit == record.flops
    assert _cost_of(cuda.function) == _cost_of(amx.function)


def test_matmul_layout_changes_only_the_per_unit_work() -> None:
    records = []
    functions = {function.name: function for function in _MatmulLayouts.functions}
    for function in (functions["plain"], functions["split"], functions["broadcast"]):
        result = analyze(_MatmulLayouts, function, analysis="roofline")
        call = _calls(result.function)[-1]
        cost = get_metadata(call, ComputeCostMetadata)
        bound = get_metadata(call, RooflineMetadata)
        assert cost is not None and bound is not None
        records.append((cost, bound, get_metadata(_calls(result.function)[-1], TrafficMetadata)))

    (
        (plain_cost, plain_bound, plain_moved),
        (split_cost, split_bound, split_moved),
        (broadcast_cost, broadcast_bound, broadcast_moved),
    ) = records
    total_flops = plain_cost.flops[0]
    assert split_cost.flops == broadcast_cost.flops == (total_flops,)
    assert broadcast_cost.flops_per_unit == plain_cost.flops
    split_per_unit = split_cost.flops_per_unit[0]
    assert (split_per_unit[0], split_per_unit[1] * _MatmulLayouts.topologies[0].size) == (
        total_flops[0],
        total_flops[1],
    )
    assert split_moved.whole == plain_moved.whole == broadcast_moved.whole
    assert split_bound == plain_bound == broadcast_bound


def test_function_call_carries_the_callee_per_unit_work() -> None:
    entry = _NestedSplitAdd.entry_function()
    result = analyze(_NestedSplitAdd, entry, analysis=("compute-cost", "memory"))

    record = get_metadata(_calls(result.function)[-1], ComputeCostMetadata)
    assert record is not None
    assert record.flops == (("f32", 256),)
    assert record.flops_per_unit == (("f32", 64),)


def test_a_mesh_naming_two_levels_is_segmented_where_its_axes_land() -> None:
    """One Mesh may name both levels, and its axes say where the boundary is.

    A warp-specialized kernel is placed per CTA and per lane at once, and one
    mesh naming both states that in one place. Its axes are handed to the levels
    left to right, so the boundary is where their extents multiply to the CTA
    count; the CTA projection reads that segment and the lane axes divide out.
    An axis that straddles the boundary is refused rather than halved: four
    positions across two CTAs of two lanes could be either half of it.
    """
    with pytest.raises(AnalysisError, match="do not land on the boundary of level 'cta'"):
        _run(_multi_topology_mesh, "compute-cost")

    placed, function = _run(_segmented_mesh, "performance")
    assert placed.executed[-1] == "performance"
    calls = _calls(function)
    assert calls and all(
        get_metadata(call, ExecutionDomainMetadata).at("cta") is not None for call in calls
    )
    cta = IrTopology("cta", 2)
    assert _mesh_image(
        get_metadata(calls[-1], ExecutionDomainMetadata).at("cta"), cta
    ) == frozenset((0, 1))


def test_saying_where_the_lanes_are_does_not_move_the_ctas() -> None:
    """Three spellings of one placement, held to the same answer.

    A warp-specialized kernel is placed per CTA and per lane. It can say only the
    CTA half, or say both on one Mesh naming both levels, or say both as a lane
    Mesh nested inside a CTA Mesh. Those are three ways of writing one program,
    so the CTA participants and the predicted time have to come out the same:
    what the lanes do is what happens inside a CTA, and a reading that let it
    change which CTAs run would be pricing the spelling rather than the program.
    """
    answers = {}
    for name in ("Levels", "LevelsOnOneMesh", "LevelsNested"):
        root = getattr(findings, name)
        result = analyze(
            root,
            root.entry_function(),
            analysis=("compute-cost", "memory", "roofline", "performance"),
            level="cta",
        )
        summary = get_metadata(result.function, PerformanceSummaryMetadata)
        assert summary is not None
        places = {
            _mesh_image(
                get_metadata(call, ExecutionDomainMetadata).at("cta"),
                root.resolve_topology("cta"),
            )
            for call in _calls(result.function)
        }
        assert len(places) == 1
        answers[name] = (summary.timeline.end_ns, next(iter(places)))

    assert len(set(answers.values())) == 1, answers
    (_ns, placement), *_ = answers.values()
    assert placement == frozenset(range(128))


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
    rotated_result = analyze(_Rotated, rotated, analysis=("compute-cost", "memory"))
    rotation = get_metadata(
        next(
            call
            for call in _calls(rotated_result.function)
            if type(call.target).__name__ == "RoPE"
        ),
        ComputeCostMetadata,
    )
    assert rotation is not None

    assert rotation.flops == (("f32", 3 * 2 * 64),)

    allocated = _Allocated.entry_function()
    allocated_result = analyze(_Allocated, allocated, analysis=("compute-cost", "memory"))
    zeroing = next(
        call
        for call in _calls(allocated_result.function)
        if type(call.target).__name__ == "Zeros"
    )
    zeros = get_metadata(zeroing, ComputeCostMetadata)
    assert zeros is not None
    assert zeros.flops == ()
    traffic = get_metadata(zeroing, TrafficMetadata).at("gmem")
    assert traffic.write == 64 * 4
    assert traffic.read == 0


def test_index_select_reads_the_rows_it_names_and_not_the_table() -> None:
    result, function = _run(_IndexSelected, ("compute-cost", "memory"))

    record = get_metadata(_calls(function)[-1], ComputeCostMetadata)
    record_moved = get_metadata(_calls(function)[-1], TrafficMetadata)
    assert record is not None
    traffic = record_moved.at("gmem")
    rows = 4 * 64 * 4
    assert traffic.read == rows + 4 * 4
    assert traffic.write == rows

    payload = json.loads(render_json(report(result)))
    table, indices, produced = payload["calls"][-1]["traffic"]["operands"]
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
        _run(_wide_grid, "memory")


def test_memory_holds_parameters_resident_past_their_last_reader() -> None:
    """A function cannot reclaim storage its caller handed to it."""
    entry = _WeightedAdd.entry_function()
    result = analyze(_WeightedAdd, entry, analysis="memory")

    record = get_metadata(result.function, MemoryMetadata)
    assert record is not None
    held = [item for item in record.lifetimes if item.persistent]
    ordinary = [item for item in record.lifetimes if not item.persistent]
    assert [item.binding for item in held] == ["source", "weight"]
    assert ordinary, "body allocations are still measured by their live ranges"
    assert all(
        item.last_used_at == max(row.last_used_at for row in record.lifetimes) for item in held
    )






def test_a_view_keeps_the_buffer_it_reads_alive() -> None:
    """A base is live while anything reading through it still has a reader."""
    held = next(function for function in _MovementCosts.functions if function.name == "held")
    result = analyze(_MovementCosts, held, analysis="memory")

    made, _window, consumer = _calls(result.function)
    record = get_metadata(result.function, MemoryMetadata)
    assert record is not None
    bindings = {item.binding: item for item in record.lifetimes}
    assert set(bindings) == {"source", binding_name(made), binding_name(consumer)}
    assert bindings[binding_name(made)].last_used_at == bindings[
        binding_name(consumer)
    ].defined_at


def test_reshard_costs_no_op_and_cross_storage_copy() -> None:
    no_op, no_op_function = _run(_no_op_reshard, ("compute-cost", "memory"))
    assert no_op.executed == ("compute-cost", "memory")
    no_op_cost = get_metadata(_calls(no_op_function)[0], ComputeCostMetadata)
    no_op_cost_moved = get_metadata(_calls(no_op_function)[0], TrafficMetadata)
    assert no_op_cost is not None
    assert no_op_cost_moved.whole == ()
    assert no_op_cost_moved.operands == (TrafficBytes(), TrafficBytes())

    copied = analyze(SquareCuda, SquareCuda.entry_function(), analysis=("compute-cost", "memory"))
    copied_in = _calls(copied.function)[0]
    copied_cost = get_metadata(copied_in, ComputeCostMetadata)
    copied_cost_moved = get_metadata(copied_in, TrafficMetadata)
    assert copied_cost is not None
    moved = 168 * 4
    assert copied_cost_moved.whole == (
        ("gmem", TrafficBytes(read=moved)),
        ("rmem", TrafficBytes(write=moved)),
    )
    assert copied_cost_moved.operands == (
        TrafficBytes(read=moved),
        TrafficBytes(write=moved),
    )


def test_slice_costs_coordinates_but_not_the_view() -> None:
    source = Var(type=make_tensor_type((1024, 2048)), name="source")
    scalar = make_tensor_type((), DType.i64)
    starts = Tuple(
        type=TupleType(fields=(scalar, scalar)),
        elements=(Constant(type=scalar, value=0), Constant(type=scalar, value=0)),
    )
    output = make_tensor_type((256, 2048))
    call = Call(
        type=output,
        target=Slice(sizes=(256, 2048), strides=(1, 1)),
        args=(source, starts),
    )

    cost = CostEvaluator(CostContext(selected_output_type=output)).visit_Call(call)

    assert cost.traffic == (
        TrafficBytes(),
        TrafficBytes(read=tensor_bytes(starts.type)),
        TrafficBytes(),
    )


def test_unmaterialized_shape_values_are_not_charged_as_attributes() -> None:
    function = _UnmaterializedValueCosts.entry_function()
    zeros = _calls(function)[-1]
    shape_value = zeros.target.type.shape[0]
    assert shape_value.type.storage is StorageKind.UMAT
    assert shape_value not in zeros.args
    assert zeros.type.storage is StorageKind.UMAT
    traffic, operands = _call_movement(zeros, Cost({}, (TrafficBytes(write=16),)))
    assert operands == (TrafficBytes(write=16),)
    assert traffic == ()




def test_mixed_runtime_coordinates_are_charged_per_leaf() -> None:
    function = _RuntimeCoordinateCosts.entry_function()
    result = analyze(_RuntimeCoordinateCosts, function, analysis=("compute-cost", "memory"))

    (slice_call,) = _calls(result.function)
    starts = slice_call.args[1]
    assert tuple(str(field.storage) for field in starts.type.fields) == (
        "gmem",
        "umat",
    )
    record = get_metadata(slice_call, ComputeCostMetadata)
    record_moved = get_metadata(slice_call, TrafficMetadata)
    assert record is not None
    assert record_moved.whole == (
        ("gmem", TrafficBytes(read=8)),
        ("rmem", TrafficBytes(read=8)),
    )
    assert record_moved.operands[1] == TrafficBytes(read=16)


def test_typed_arange_operands_are_charged_at_their_declared_storage() -> None:
    function = _UnmaterializedIndexTensorCosts.entry_function()
    result = analyze(_UnmaterializedIndexTensorCosts, function, analysis=("compute-cost", "memory"))

    (binary,) = [call for call in _calls(result.function) if isinstance(call.target, Binary)]
    assert tuple(arg.type.shape for arg in binary.args) == ((1, 128), (128, 1))
    assert all(arg.type.storage is StorageKind.GMEM for arg in binary.args)
    record = get_metadata(binary, ComputeCostMetadata)
    record_moved = get_metadata(binary, TrafficMetadata)
    assert record is not None
    assert record_moved.whole == (("gmem", TrafficBytes(read=2_048, write=2_048)),)
    assert record_moved.operands == (
        TrafficBytes(read=1_024),
        TrafficBytes(read=1_024),
        TrafficBytes(write=2_048),
    )


def test_non_divisible_window_cost_is_a_full_tile_upper_bound() -> None:
    function = WindowCost.entry_function()
    result = analyze(WindowCost, function, analysis=("compute-cost", "memory"))
    adds = [call for call in _calls(result.function) if isinstance(call.target, Binary)]
    loop_cost = get_metadata(adds[-1], ComputeCostMetadata)
    root_cost = get_metadata(result.function, ComputeCostMetadata)

    assert loop_cost is not None
    assert loop_cost.flops == (("f32", 4 * 4),)
    assert root_cost is not None
    assert root_cost.flops == (("f32", 4 * 4 + 3 * 4 * 4),)


def test_a_sharded_shared_tile_is_measured_once_and_placed_once() -> None:
    """The capacity is an input fact; the claim is two tile lifetimes at once.

    Both tiles are live together because the second reads the first, so the
    claim is not something an ordering could reduce. A machine that can hold it
    places it and reports the footprint; the one that cannot refuses, and says
    what it was asked to hold rather than what one value cost.
    """
    functions = {function.name: function for function in _SharedTile.functions}
    roomy = replace(_SharedTile, target=_RoomyShared("nvidia.h200_sxm"))
    result = analyze(
        roomy,
        next(item for item in roomy.functions if item.name == "split"),
        analysis="memory",
    )

    record = get_metadata(result.function, MemoryMetadata)
    assert record is not None
    smem = next(item for item in record.footprint if item.level == "smem")
    largest_tile = max(item.bytes for item in record.lifetimes if item.level == "smem")
    assert smem.peak_bytes == 2 * largest_tile == smem.capacity_bytes
    assert record.allocation == AllocationMetadata(solver_status="optimal")

    with pytest.raises(AnalysisError, match=r"'smem' holds 422400 B at one point"):
        analyze(_SharedTile, functions["split"], analysis="memory")
    with pytest.raises(AnalysisError, match=r"27878400 B in smem"):
        analyze(_SharedTile, functions["broadcast"], analysis="memory")


def test_memory_footprints_follow_the_owner_recorded_by_the_target() -> None:
    matmul = next(fn for fn in _MatmulLayouts.functions if fn.name == "split")
    result = analyze(_MatmulLayouts, matmul, analysis="memory")
    gmem = get_metadata(result.function, MemoryMetadata)
    assert gmem is not None
    gmem_lifetimes = {item.binding: item.bytes for item in gmem.lifetimes if item.level == "gmem"}
    assert "v0" not in gmem_lifetimes
    assert gmem_lifetimes["lhs"] == tensor_bytes(matmul.params[0].type)

    roomy = replace(_SharedTile, target=_RoomyShared("nvidia.h200_sxm"))
    shared = next(fn for fn in roomy.functions if fn.name == "split")
    result = analyze(roomy, shared, analysis="memory")
    cta_owned = get_metadata(result.function, MemoryMetadata)
    assert cta_owned is not None
    local_bytes = next(
        item.bytes
        for item in cta_owned.lifetimes
        if item.binding == "v0" and item.level == "smem"
    )
    assert local_bytes == tensor_bytes(shared.params[0].type) // _SharedTile.topologies[0].size

    thread_shared = _entry(_modest_shared)
    result = analyze(_modest_shared, thread_shared, analysis="memory")
    still_cta_owned = get_metadata(result.function, MemoryMetadata)
    assert still_cta_owned is not None
    assert (
        next(
            item.bytes
            for item in still_cta_owned.lifetimes
            if item.binding == "v0" and item.level == "smem"
        )
        == tensor_bytes(thread_shared.params[0].type)
    )

    registers = _entry(_thread_sharded)
    result = analyze(_thread_sharded, registers, analysis="memory")
    thread_owned = get_metadata(result.function, MemoryMetadata)
    assert thread_owned is not None
    assert (
        next(
            item.bytes
            for item in thread_owned.lifetimes
            if item.binding == "v0" and item.level == "rmem"
        )
        == 8
    )


def test_analysis_snapshot_drift_sentinel() -> None:
    """Intentional current-output snapshot; change it only with model review."""
    functions = {function.name: function for function in _MatmulLayouts.functions}
    matmul_records = []
    split_result = None
    for function in (functions["plain"], functions["split"], functions["broadcast"]):
        result = analyze(_MatmulLayouts, function, analysis="roofline")
        if function is functions["split"]:
            split_result = analyze(_MatmulLayouts, function, analysis="memory")
        call = _calls(result.function)[-1]
        matmul_records.append(
            (
                get_metadata(call, ComputeCostMetadata),
                get_metadata(call, RooflineMetadata),
            )
        )
    plain_cost, plain_bound = matmul_records[0]
    split_cost, _split_bound = matmul_records[1]

    roomy = replace(_SharedTile, target=_RoomyShared("nvidia.h200_sxm"))
    shared = next(function for function in roomy.functions if function.name == "split")
    shared_result = analyze(roomy, shared, analysis="memory")
    shared_record = get_metadata(shared_result.function, MemoryMetadata)
    assert shared_record is not None
    shared_smem = shared_record.level("smem")
    assert shared_smem is not None

    assert split_result is not None
    gmem = get_metadata(split_result.function, MemoryMetadata)
    assert gmem is not None
    gmem_lifetimes = {
        item.binding: item.bytes for item in gmem.lifetimes if item.level == "gmem"
    }

    modest = _entry(_modest_shared)
    modest_result = analyze(_modest_shared, modest, analysis="memory")
    modest_record = get_metadata(modest_result.function, MemoryMetadata)
    assert modest_record is not None
    modest_local = next(
        item.bytes
        for item in modest_record.lifetimes
        if item.binding == "v0" and item.level == "smem"
    )

    placed = []
    placed_traffic = []
    for dims in ({"ctx": 1024, "seq": 1}, {"ctx": 1, "seq": 1024}):
        result = analyze(
            PrefillDecodeAttention,
            PrefillDecodeAttention.entry_function(),
            analysis="roofline",
            dims=dims,
        )
        record = get_metadata(result.function, RooflineMetadata)
        assert record is not None
        placed.append(record.ideal_ns)
        placed_traffic.append(report(result)["totals"]["traffic"])

    flash = analyze(
        FlashSplitKDecode,
        FlashSplitKDecode.entry_function(),
        analysis="roofline",
        dims={"ctx": 4096},
    )
    flash_slices = tuple(
        get_metadata(expr, TrafficMetadata).whole
        for expr in postorder(flash.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, Slice)
    )

    snapshot = {
        "matmul_flops": plain_cost.flops,
        "matmul_split_flops_per_unit": split_cost.flops_per_unit,
        "matmul_ideal_ns": plain_bound.ideal_ns,
        "shared_peak_bytes": shared_smem.peak_bytes,
        "shared_largest_tile_bytes": max(
            item.bytes for item in shared_record.lifetimes if item.level == "smem"
        ),
        "gmem_lhs_bytes": gmem_lifetimes["lhs"],
        "modest_shared_bytes": modest_local,
        "placed_ideal_ns": tuple(placed),
        "placed_traffic": tuple(placed_traffic),
        "flash_split_traffic": report(flash)["totals"]["traffic"],
        "flash_split_offset_slice_traffic": flash_slices,
    }
    assert snapshot == {
        "matmul_flops": (("bf16", 28_547_481_600),),
        "matmul_split_flops_per_unit": (("bf16", 216_268_800),),
        "matmul_ideal_ns": 28_851,
        "shared_peak_bytes": 422_400,
        "shared_largest_tile_bytes": 211_200,
        "gmem_lhs_bytes": 4_325_376,
        "modest_shared_bytes": 4_096,
        "placed_ideal_ns": (3_501, 65_194),
        "placed_traffic": (
            {
                "gmem": {"read": 8_407_240, "write": 8_396_936},
                "rmem": {"read": 672, "write": 0},
                    "smem": {"read": 21_669_568, "write": 21_432_000},
            },
            {
                "gmem": {"read": 4_472_832, "write": 8_536_064},
                "rmem": {"read": 960, "write": 0},
                    "smem": {"read": 794_492_928, "write": 484_114_432},
            },
        ),
        "flash_split_traffic": {
            "gmem": {"read": 2_099_264, "write": 2_048},
            "rmem": {"read": 336, "write": 104},
            "smem": {"read": 11_899_776, "write": 11_578_752},
        },
        "flash_split_offset_slice_traffic": (
            (
                ("gmem", TrafficBytes(read=0, write=0)),
                    ("rmem", TrafficBytes(read=32, write=0)),
            ),
            (
                ("gmem", TrafficBytes(read=0, write=0)),
                    ("rmem", TrafficBytes(read=32, write=0)),
            ),
        ),
    }


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

    loop = next(
        expr
        for expr in postorder(entry.body)
        if isinstance(expr, GridRegionExpr)
    )
    footprint = get_metadata(loop, LoopFootprintMetadata)
    assert footprint is not None and footprint.known
    assert len(footprint.footprints) == 2
    assert all(
        item.device_bytes == item.bytes == 64_000_000
        for item in footprint.footprints
    )

    rendered = render_analysis(result)
    assert rendered.data["loops"][0]["cache-pressure"] == [
        {
            "cache_level": "l2",
            "backing_level": "gmem",
            "device_bytes": 128_000_000,
            "capacity_bytes": 50_000_000,
            "status": "exceeds",
        }
    ]
    lines = render_text(rendered).splitlines()
    assert [f"# advisory={json.dumps(note)}" for note in record.advisories] == [
        line for line in lines if line.startswith("# advisory")
    ]


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
    call = _calls(result.function)[-1]
    cost = get_metadata(call, ComputeCostMetadata)
    bound = get_metadata(call, RooflineMetadata)
    assert cost is not None and bound is not None

    facts = CudaTarget("nvidia.h200_sxm").get_facts(ThroughputFacts)
    rate = facts.peak_for(DType.f32)
    assert rate is not None
    expected = -(-cost.flops[0][1] * 1_000_000_000 // rate)
    assert bound.compute_ns == expected

    mixed = _MixedPrecision.entry_function()
    mixed_result = analyze(_MixedPrecision, mixed, analysis="roofline")
    whole = get_metadata(mixed_result.function, RooflineMetadata)
    assert whole is not None
    per_call = [
        record
        for call in _calls(mixed_result.function)
        if (record := get_metadata(call, RooflineMetadata)) is not None
    ]
    assert len(per_call) > 1
    assert whole.compute_ns < sum(item.compute_ns for item in per_call)
    assert whole.ideal_ns == max(whole.compute_ns, whole.memory_ns)


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


def test_performance_refuses_unplaced_results_before_dependencies_run(monkeypatch) -> None:
    """A topology declaration does not place a value or admit dependency writes."""
    compute = analyze(_wide_grid, _entry(_wide_grid), analysis="compute-cost")
    footprint = analyze(_wide_grid, _entry(_wide_grid), analysis="memory")
    roofline = analyze(_wide_grid, _entry(_wide_grid), analysis="roofline")
    assert get_metadata(compute.function, ComputeCostMetadata) is not None
    assert get_metadata(footprint.function, MemoryMetadata) is not None
    assert get_metadata(roofline.function, RooflineMetadata) is not None

    placed, function = _run(_placed_wide_grid, "performance")
    assert placed.executed == ("compute-cost", "memory", "performance")
    assert [
        type(call.target).__name__
        for call in _calls(function)
        if get_metadata(call, PerformanceMetadata) is not None
    ] == ["Binary"]

    ran = False

    def unexpected(*_args, **_kwargs):
        nonlocal ran
        ran = True

    monkeypatch.setattr(compute_cost, "analyze_compute_cost", unexpected)
    with pytest.raises(
        AnalysisError,
        match=r"test_analysis_families.py:.*has no cta execution domain",
    ):
        _run(_wide_grid, "performance")
    assert not ran


def test_mega_kernel_preserves_placement_costs_and_dependency_order() -> None:
    """The expanded placed program keeps its slices, costs, and dependency order.

    Two roots are asked for, and the four families arrive: the bound and the
    prediction share compute-cost, and asking for both measures it once.
    """
    result = analyze(
        MoEMegaKernel,
        MoEMegaKernel.entry_function(),
        analysis=("roofline", "performance"),
    )
    assert result.executed == ("compute-cost", "memory", "roofline", "performance")
    calls = _calls(result.function)
    assert len(calls) == 7
    assert all(not isinstance(call.target, Function) for call in calls)

    for index, shape, offset in (
        (0, (120,), 0),
        (1, (120,), 0),
        (3, (12,), 120),
        (4, (12,), 120),
    ):
        call_type = calls[index].type
        assert isinstance(call_type, TensorType)
        assert isinstance(call_type.layout, ShardLayout)
        layout = call_type.layout.mesh.layout
        assert isinstance(layout, ComposedLayout)
        assert layout.outer is not None
        assert (layout.outer.shape, layout.offset) == (shape, offset)

    placements = _call_placements(MoEMegaKernel, result.function, "cta")
    prepared = set(placements.values())

    assert frozenset(range(120)) in prepared
    assert frozenset(range(120, 132)) in prepared
    assert frozenset(range(132)) in prepared

    views = [call for call in calls if type(call.target).__name__ == "Reshard"]
    assert len(views) == 4
    for view in views:
        view_cost = get_metadata(view, ComputeCostMetadata)
        view_cost_moved = get_metadata(view, TrafficMetadata)
        assert view_cost is not None
        assert view_cost_moved.whole == ()
        assert view_cost_moved.per_unit == ()
        assert view_cost_moved.operands == (TrafficBytes(), TrafficBytes())

    call_costs = [get_metadata(call, ComputeCostMetadata) for call in calls]
    call_moved = [get_metadata(call, TrafficMetadata) for call in calls]
    assert all(call_cost is not None for call_cost in call_costs)
    summed = TrafficBytes(
        read=sum(moved.at("gmem").read for moved in call_moved),
        write=sum(moved.at("gmem").write for moved in call_moved),
    )
    root_cost = get_metadata(result.function, ComputeCostMetadata)
    root_cost_moved = get_metadata(result.function, TrafficMetadata)
    memory = get_metadata(result.function, MemoryMetadata)
    roofline = get_metadata(result.function, RooflineMetadata)
    assert root_cost is not None
    assert memory is not None
    assert roofline is not None
    assert dict(root_cost.flops) == {"f32": 23_040}
    assert dict(root_cost.flops) == {
        "f32": sum(dict(call_cost.flops).get("f32", 0) for call_cost in call_costs)
    }
    assert root_cost_moved.at("gmem") == summed == TrafficBytes(
        read=122_880,
        write=92_160,
    )
    assert get_metadata(result.function, TrafficMetadata).at("gmem") == summed
    assert (roofline.memory_ns, roofline.ideal_ns, roofline.bound_by) == (
        45,
        45,
        "memory",
    )

    positive_per_unit = [
        dict(call_cost.flops_per_unit)["f32"]
        for call_cost in call_costs
        if call_cost.flops_per_unit
    ]
    assert positive_per_unit == [64, 640, 7_680]
    relu = next(call for call in calls if type(call.target).__name__ == "ReLU")
    cost = get_metadata(relu, ComputeCostMetadata)
    cost_moved = get_metadata(relu, TrafficMetadata)
    assert cost is not None
    assert cost_moved.whole == (("gmem", TrafficBytes(read=30_720, write=30_720)),)
    assert cost_moved.per_unit == (("gmem", TrafficBytes(read=256, write=256)),)
    assert cost_moved.per_unit_at("gmem") == TrafficBytes(read=256, write=256)
    assert (
        "traffic=gmem:r30720/w30720@r256/w256"
        in render_comment(cost_moved, opt_in=frozenset({"operands"}))
    )

    records = tuple(get_metadata(call, PerformanceMetadata) for call in calls)
    assert [index for index, record in enumerate(records) if record is not None] == [
        1,
        4,
        6,
    ]
    routed, shared, join = (records[index].timeline for index in (1, 4, 6))

    assert [
        (record.start_ns, record.end_ns) for record in (routed, shared, join)
    ] == [(0, 15), (0, 141), (141, 2_676)]

    routed_positions = placements[id(calls[1])]
    shared_positions = placements[id(calls[4])]
    assert routed_positions.isdisjoint(shared_positions)
    assert routed.start_ns < shared.end_ns and shared.start_ns < routed.end_ns

    whole_positions = placements[id(calls[2])]
    assert whole_positions == placements[id(calls[5])] == frozenset(range(132))
    assert whole_positions != shared_positions
    assert whole_positions & shared_positions
    assert join.start_ns >= max(routed.end_ns, shared.end_ns)

    summary = get_metadata(result.function, PerformanceSummaryMetadata)
    assert summary == PerformanceSummaryMetadata(
        timeline=TimelineMetadata(end_ns=2_676),
        waves=1,
    )


def test_performance_scales_one_local_schedule_by_root_capacity() -> None:
    """Root geometry and capacity scale waves, never occurrence intervals."""
    results = [
        analyze(
            _ScalableGrid,
            _ScalableGrid.entry_function(),
            analysis="performance",
            dims={"grid": extent},
        )
        for extent in (132, 265)
    ]
    constrained = replace(
        _ScalableGrid,
        target=_ConstrainedCudaTarget("nvidia.h200_sxm"),
    )
    results.append(
        analyze(
            constrained,
            constrained.entry_function(),
            analysis="performance",
            dims={"grid": 265},
        )
    )

    intervals = [
        tuple(get_metadata(call, PerformanceMetadata) for call in _calls(result.function))
        for result in results
    ]
    summaries = [
        get_metadata(result.function, PerformanceSummaryMetadata) for result in results
    ]
    assert intervals[0] == intervals[1] == intervals[2]
    assert all(summary is not None for summary in summaries)
    first, wider, constrained_summary = summaries
    assert first is not None and wider is not None and constrained_summary is not None
    assert (first.waves, wider.waves, constrained_summary.waves) == (1, 3, 5)
    assert (
        first.timeline.end_ns,
        wider.timeline.end_ns,
        constrained_summary.timeline.end_ns,
    ) == (15, 45, 75)
    assert {summary.timeline.start_ns for summary in summaries} == {0}

    data = report(results[1])
    assert data["function_records"]["performance"] == {
        "timeline": {"start_ns": 0, "end_ns": 45, "trips": 1, "stride_ns": 0},
        "waves": 3,
    }
    assert all(
        set(row["performance"]["timeline"]) == {"start_ns", "end_ns", "trips", "stride_ns"}
        for row in data["calls"]
    )


def test_placement_projection_rejects_the_wrong_level_and_invalid_images() -> None:
    with pytest.raises(AnalysisError, match="has no cta execution domain"):
        analyze(_thread_sharded, _entry(_thread_sharded), analysis="performance")

    thread_only = make_tensor_type(
        (8,),
        layout=ShardLayout(
            Layout((8,), (1,)),
            (Broadcast(),),
            IrMesh((IrTopology("thread", 4),), Layout((4,), (1,))),
        ),
    )
    with pytest.raises(
        AnalysisError,
        match=r"selected topology level 'cta'.*placed at level.s. 'thread'",
    ):
        _result_placement(thread_only, IrTopology("cta", 8))

    cta = IrTopology("cta", 8)
    outside = IrMesh(
        (cta,),
        ComposedLayout(None, 7, Layout((2,), (1,))),
    )
    with pytest.raises(AnalysisError, match=r"positions \[8\].*outside.*\[0, 8\)"):
        _mesh_image(outside, cta)

    dynamic = replace(
        outside,
        layout=ComposedLayout(None, 0, Layout((DimVar("mesh_n", 1, 9),), (1,))),
    )
    with pytest.raises(AnalysisError, match="not projectable"):
        _mesh_image(dynamic, cta)

    strict_plain = IrMesh((cta,), Layout((4,), (1,)))
    with pytest.raises(AnalysisError, match="must describe the full selected domain"):
        _mesh_image(strict_plain, cta)

    left = IrMesh((cta,), ComposedLayout(None, 0, Layout((4,), (1,))))
    right = IrMesh((cta,), ComposedLayout(None, 4, Layout((4,), (1,))))
    tuple_type = TupleType(
        (
            make_tensor_type((8,), layout=ShardLayout(Layout((8,), (1,)), (Broadcast(),), left)),
            make_tensor_type((8,), layout=ShardLayout(Layout((8,), (1,)), (Broadcast(),), right)),
        )
    )
    with pytest.raises(AnalysisError, match="tuple result leaves carry different"):
        _result_placement(tuple_type, cta)


def test_nested_levels_project_independently_and_broadcast_counts_as_placed() -> None:
    cta = IrTopology("cta", 2)
    thread = IrTopology("thread", 4)
    thread_layout = ShardLayout(
        Layout((8,), (1,)),
        (Broadcast(),),
        IrMesh((thread,), Layout((4,), (1,))),
    )
    nested = ShardLayout(
        thread_layout,
        (Broadcast(),),
        IrMesh((cta,), Layout((2,), (1,))),
    )
    type_ = TensorType((8,), DType.f32, nested, storage="rmem")

    assert _result_placement(type_, cta) == frozenset((0, 1))
    assert _result_placement(type_, thread) == frozenset(range(4))


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


def test_a_view_and_its_base_are_one_rectangle_to_place() -> None:
    """Shared memory holds the tile once, however many names read it.

    The same room that holds the tile and the result holds a view of the tile as
    well, because the view is the tile. Giving the view an address of its own
    would need half again as much and this program would not fit.
    """
    roomy = replace(_SharedTile, target=_RoomyShared("nvidia.h200_sxm"))
    viewed = next(function for function in roomy.functions if function.name == "viewed")

    result = analyze(roomy, viewed, analysis=("memory", "performance"))

    tile, _view, _sum = _calls(result.function)
    record = get_metadata(result.function, MemoryMetadata)
    assert record is not None
    assert record.level("smem").peak_bytes == _RoomyShared.levels["smem"]
    assert [item.binding for item in record.lifetimes if item.level == "smem"] == [
        binding_name(tile),
        binding_name(_sum),
    ]
    assert get_metadata(result.function, PerformanceSummaryMetadata) is not None






def test_a_machine_that_places_nothing_is_not_a_machine_that_fits() -> None:
    """No allocation record and a settled one are different answers.

    This program never names the level its machine runs work at, so there is no
    unit for a buffer to belong to and nothing to place it against: nothing was
    decided and nothing is claimed, which is not the same as having looked and
    found room. The buffers are still measured, because measuring them never
    needed an address.
    """
    unplaceable = analyze(_NoParallelLevel, _NoParallelLevel.entry_function(), analysis="memory")
    record = get_metadata(unplaceable.function, MemoryMetadata)
    assert record is not None and record.allocation is None
    assert record.lifetimes

    placed = analyze(_CudaAdd, _CudaAdd.entry_function(), analysis="memory")
    assert get_metadata(placed.function, MemoryMetadata).allocation == AllocationMetadata(
        solver_status="optimal"
    )


def test_a_view_with_no_layout_still_runs_where_it_was_written() -> None:
    """Where an occurrence runs is where it was authored, not what it produced.

    Three of this program's occurrences hand back a view carrying no layout at
    all, and a reading that asked the result type where the work ran had to
    refuse the whole program over them. The Mesh they were written inside
    answers instead, so every occurrence is placed and the program is priced.
    Placing buffers stays the narrower question it was: each value that holds
    bytes says where it sits, down to the shared-memory tiles the loop keeps.
    """
    case = next(item for item in CORPUS if item.id == "access_footprint.qkv")
    selected = next(item for item in case.analyze if item.selector == "qkv_projection")
    owner, function = case.resolve(case.build(), selected.selector)
    result = analyze(owner, function, analysis="memory", dims=selected.dims)
    calls = [
        call
        for call in _calls(result.function)
        if not isinstance(call.target, Function)
    ]
    topology = owner.resolve_topology("cta")
    viewed = [call for call in calls if _cannot_be_placed_by_type(call, topology)]
    assert [type(call.target).__name__ for call in viewed] == [
        "Slice",
        "Slice",
        "InsertSlice",
    ]
    assert len(_call_placements(owner, result.function, "cta")) == len(calls)

    record = get_metadata(result.function, MemoryMetadata)
    assert record.allocation == AllocationMetadata(solver_status="optimal")
    assert any(item.level == "smem" for item in record.lifetimes)
    priced = analyze(owner, function, analysis="performance", dims=selected.dims)
    assert get_metadata(priced.function, PerformanceSummaryMetadata) is not None


def _cannot_be_placed_by_type(call: Call, topology: IrTopology) -> bool:
    """Whether this occurrence's own result type says nothing about where it ran."""
    try:
        _result_placement(call.type, topology)
    except AnalysisError:
        return True
    return False


def test_the_innermost_scope_naming_a_level_is_the_one_that_ran_it() -> None:
    """A nest of Meshes answers about the level asked for, from the inside out.

    An occurrence written inside a thread Mesh inside a CTA Mesh ran on both,
    and which one is meant depends entirely on the level being asked about. The
    stack is kept as authored so each question reaches its own answer, and a
    level nobody named has none rather than the nearest thing to one.
    """
    cta = IrMesh((IrTopology("cta", 2),), Layout((2,), (1,)))
    thread = IrMesh((IrTopology("thread", 4),), Layout((4,), (1,)))
    inner = IrMesh((IrTopology("cta", 8),), Layout((8,), (1,)))
    domain = ExecutionDomainMetadata((cta, thread, inner))

    assert domain.at("cta") is inner
    assert domain.at("thread") is thread
    assert domain.at("warp") is None
    assert ExecutionDomainMetadata().at("cta") is None


def test_a_result_cannot_be_placed_where_the_work_that_made_it_never_ran() -> None:
    """Both answers are kept, and where they disagree the program is refused.

    A layout says where a result's bytes were put and a Mesh says where the work
    was written, so where a result carries the selected level at all the two are
    claims about one thing and cannot differ. Taking either one silently would
    be picking which half of a contradiction to believe.
    """
    cta = IrTopology("cta", 8)
    left = IrMesh((cta,), ComposedLayout(None, 0, Layout((4,), (1,))))
    right = IrMesh((cta,), ComposedLayout(None, 4, Layout((4,), (1,))))
    placed = make_tensor_type(
        (8,), layout=ShardLayout(Layout((8,), (1,)), (Broadcast(),), right)
    )
    call = Call(
        type=placed,
        target=Transpose(perm=(0,)),
        args=(Var(type=placed, name="x"),),
        metadata=(ExecutionDomainMetadata((left,)),),
    )
    with pytest.raises(AnalysisError, match="cannot be placed where the work"):
        _execution_placement(call, cta)

    agreeing = replace(call, metadata=(ExecutionDomainMetadata((right,)),))
    assert _execution_placement(agreeing, cta) == _result_placement(placed, cta)


def test_an_execution_domain_is_restated_at_the_extents_it_is_asked_at() -> None:
    """A program authored over a range says where it runs at the size chosen.

    The Mesh a Call was written inside is a value like the types beside it, and
    this one names a CTA count derived from an authored dimension. Carried
    through a specialization untouched it would leave a concrete program whose
    own account of where it runs is still a range, which is a domain with no
    positions to count.
    """
    result = analyze(
        DerivedPrefill,
        DerivedPrefill.entry_function(),
        analysis="performance",
        dims={"prefill_n": 64, "topology_only": 128},
    )
    scopes = [
        get_metadata(call, ExecutionDomainMetadata) for call in _calls(result.function)
    ]
    assert scopes and all(scope is not None for scope in scopes)
    assert {scope.at("cta").topologies[0].size for scope in scopes} == {8}
    assert {size(scope.at("cta").layout) for scope in scopes} == {8}
    assert all(
        get_metadata(call, PerformanceMetadata) is not None
        or _is_structural_occurrence(get_metadata(call, ComputeCostMetadata))
        for call in _calls(result.function)
    )


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


def test_only_addressable_levels_are_placed_by_the_solver() -> None:
    """Registers hold what they hold; this model does not place them.

    The two register values here are live at once and each is as big as the
    level the target states, so a solver that packed registers would call this
    impossible. It is not: allocation stops at the levels this model addresses.
    """
    narrow = replace(_PricingBoundary, target=_NarrowRegisters("nvidia.h200_sxm"))
    function = next(item for item in narrow.functions if item.name == "unpriced_only")

    result = analyze(narrow, function, analysis="performance")

    summary = get_metadata(result.function, PerformanceSummaryMetadata)
    assert summary is not None
    assert get_metadata(result.function, MemoryMetadata).allocation.solver_status == "optimal"
    baseline = analyze(
        _PricingBoundary,
        next(item for item in _PricingBoundary.functions if item.name == "unpriced_only"),
        analysis="performance",
    )
    assert summary.timeline == get_metadata(
        baseline.function, PerformanceSummaryMetadata
    ).timeline


def test_time_comes_from_the_rates_the_target_states_and_from_nothing_else() -> None:
    """Movement at a level with no published bandwidth is recorded, not timed.

    The target states one bandwidth, for one level. Movement anywhere else is
    still worth reporting -- it is what the program does -- but it is nobody's
    service in this model, so it neither adds time nor refuses the program.
    Work whose dtype has no rate is the other case: it would be timed if it
    could be, so pricing it as free would be a number with a hole in it.
    """
    functions = {function.name: function for function in _PricingBoundary.functions}
    target = _PricingBoundary.resolve_target()
    throughput = target.get_facts(ThroughputFacts)
    services = target.get_facts(PerformanceServiceFacts)

    result = analyze(_PricingBoundary, functions["unpriced_only"], analysis="performance")
    priced = _calls(result.function)
    costs = [get_metadata(call, ComputeCostMetadata) for call in priced]
    moves = [get_metadata(call, TrafficMetadata) for call in priced]
    assert any(moved.per_unit_at("rmem").total_bytes for moved in moves)
    assert all(
        _local_duration_ns(record, throughput, services, moved=bytes_, level="cta")
        == _local_duration_ns(
            record,
            throughput,
            services,
            moved=_priced_levels_only(bytes_, throughput),
            level="cta",
        )
        for record, bytes_ in zip(costs, moves)
    )

    result = analyze(_PricingBoundary, functions["mixed"], analysis="performance")
    mixed = next(
        call
        for call in _calls(result.function)
        if get_metadata(call, TrafficMetadata).per_unit_at("gmem").total_bytes
        and get_metadata(call, TrafficMetadata).per_unit_at("rmem").total_bytes
    )
    work = get_metadata(mixed, ComputeCostMetadata)
    moved = get_metadata(mixed, TrafficMetadata)
    assert _local_duration_ns(
        work, throughput, services, moved=moved, level="cta"
    ) == _local_duration_ns(
        work,
        throughput,
        services,
        moved=_priced_levels_only(moved, throughput),
        level="cta",
    )

    computed = next(
        record
        for record in (
            get_metadata(call, ComputeCostMetadata) for call in _calls(result.function)
        )
        if any(value for _dtype, value in record.flops_per_unit)
    )
    with pytest.raises(
        AnalysisError,
        match=r"^performance: target states no one-unit throughput for "
        r"dtype 'f32' at 'cta'$",
    ):
        _local_duration_ns(computed, throughput, replace(services, unit_flops=()), level="cta")
    assert _local_duration_ns(
        work, throughput, replace(services, unit_ops=()), moved=moved, level="cta"
    ) == _local_duration_ns(work, throughput, services, moved=moved, level="cta"), (
        "movement at a level with no published rate was priced as service"
    )

    result = analyze(_PricingBoundary, functions["serviced_predicate"], analysis="performance")
    compared = next(
        record
        for record in (
            get_metadata(call, ComputeCostMetadata) for call in _calls(result.function)
        )
        if record.service_per_unit_of("predicate")
    )
    assert not compared.flops_per_unit
    assert _local_duration_ns(compared, throughput, services, level="cta") > 0
    with pytest.raises(
        AnalysisError,
        match=r"^performance: target states no one-unit throughput for "
        r"'predicate' work at 'cta'$",
    ):
        _local_duration_ns(compared, throughput, replace(services, unit_ops=()), level="cta")


def _priced_levels_only(
    record: TrafficMetadata, throughput: ThroughputFacts
) -> TrafficMetadata:
    """The same record with every unpriced level's traffic dropped."""
    return replace(
        record,
        per_unit=tuple(
            (level, moved)
            for level, moved in record.per_unit
            if level == throughput.bandwidth_level
        ),
    )


def test_a_zero_cost_structural_occurrence_carries_no_performance_record() -> None:
    result, entry = _run(_zero_cost_view, "performance")

    view = next(
        call for call in _calls(entry) if type(call.target).__name__ == "TupleGetItem"
    )
    assert get_metadata(view, PerformanceMetadata) is None
    rows = report(result)["calls"]
    assert rows and all(
        set(row["performance"]["timeline"]) == {"start_ns", "end_ns", "trips", "stride_ns"}
        for row in rows
    )
    assert all(
        row["performance"]["timeline"]["start_ns"] < row["performance"]["timeline"]["end_ns"]
        for row in rows
    )


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


def test_one_unit_takes_the_same_time_however_many_units_there_are() -> None:
    """A per-unit interval is one unit's, so counting the units cannot lengthen it.

    This value is held whole by every participant rather than divided among
    them, so one unit does the same work whether it is alone or one of a hundred
    and twenty. An interval that grew with the count would be the whole device's
    time wearing a unit's name, and the timeline would then say the program got
    slower for being spread out.
    """
    target = _CudaAdd.resolve_target()
    throughput = target.get_facts(ThroughputFacts)
    services = target.get_facts(PerformanceServiceFacts)

    measured = {}
    for units in (1, 120):
        spread = replace(_CudaAdd, topologies=(Topology("cta", units),))
        result = analyze(
            spread, spread.entry_function(), analysis=("compute-cost", "memory")
        )
        priced = _calls(result.function)[-1]
        work = get_metadata(priced, ComputeCostMetadata)
        moved = get_metadata(priced, TrafficMetadata)
        measured[units] = (
            work.flops_per_unit,
            moved.per_unit,
            _local_duration_ns(work, throughput, services, moved=moved, level="cta"),
        )

    alone, crowd = measured[1], measured[120]
    assert alone[0] == crowd[0] and alone[1] == crowd[1], (
        "this value was divided after all, so the two are not the same local work"
    )
    assert alone[2] > 0
    assert alone[2] == crowd[2], (
        "one unit's interval changed with how many units were declared"
    )


def test_a_local_materialise_states_bytes_this_model_puts_no_clock_on() -> None:
    """Having moved bytes and having timed work are different questions.

    NVIDIA states no register or shared-memory bandwidth, and standing an
    instruction throughput in for one prices a move as if it were arithmetic.
    This model reports those bytes and invents no rate for them, so there is no
    interval, and with no interval nothing to lay on a participant: it asks for
    no placement either, which is not the same as saying it moved nothing. A
    model that does rate them brings interval and placement back together.
    """
    _result, entry = _run(_unplaced_structural, ("compute-cost", "memory"))
    (view,) = _calls(entry)
    cost = get_metadata(view, ComputeCostMetadata)
    cost_moved = get_metadata(view, TrafficMetadata)
    assert cost is not None
    assert cost.flops_per_unit == ()
    assert cost.service_per_unit == ()
    assert cost_moved.per_unit_at("gmem").total_bytes == 0
    assert cost_moved.per_unit_at("rmem") == TrafficBytes(read=8, write=8), (
        "the bytes it moved stopped being recorded along with their rate"
    )

    target = _unplaced_structural.resolve_target()
    services = target.get_facts(PerformanceServiceFacts)
    throughput = target.get_facts(ThroughputFacts)
    assert _local_duration_ns(
        cost, throughput, services, moved=cost_moved, level="cta"
    ) == 0, "bytes at a level nobody rated were given a time anyway"
    assert _is_structural_occurrence(
        cost, cost_moved, bandwidth_level=throughput.bandwidth_level
    ), "an occurrence with nothing to time was still asked to be laid on a clock"
    bound = _cost_bound(cost, cost_moved, throughput)
    assert (bound.ideal_ns, bound.bound_by) == (0, "none"), (
        "a bound owed a nanosecond to bytes no rate was published for"
    )

    timed = analyze(
        _unplaced_structural, _entry(_unplaced_structural), analysis="performance"
    )
    (placed,) = _calls(timed.function)
    assert get_metadata(placed, PerformanceMetadata) is None, (
        "an occurrence this model puts no clock on was given an interval"
    )
    assert get_metadata(placed, TrafficMetadata).per_unit_at("rmem").total_bytes == 16



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


def test_a_memory_report_states_the_bytes_that_family_counted() -> None:
    """The family that counts the bytes is the one whose report shows them.

    Asking for memory alone asks what a program moves and where it sits, and a
    rendering of that answer that names only the footprint has dropped half of
    it. The other family is not running, so nothing here may borrow its line
    either: a report of one family states that family.
    """
    result = analyze(MoEMegaKernel, MoEMegaKernel.entry_function(), analysis="memory")
    rendering = render_analysis(result)
    text = render_text(rendering)

    assert "traffic" in rendering.data["function_records"], (
        "the memory family reported no traffic record at all"
    )
    moved = rendering.data["function_records"]["traffic"]["whole"]["gmem"]
    assert (
        f"# traffic traffic=gmem:r{moved['read']}/w{moved['write']}" in text
    ), "a memory report showed the footprint and not the bytes that made it"
    assert "# compute-cost" not in text, (
        "a family that did not run was given a line of its own"
    )
    assert "# peak-footprint" in text


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

    rendered = render_analysis(result)
    data = rendered.data

    assert data["requested"] == ["roofline"]
    assert data["executed"] == ["compute-cost", "memory", "roofline"]
    assert set(data["function_records"]) == {"roofline"}
    assert data["totals"]["flops"] == {"f32": 256}
    assert all(set(call) == {"value", "roofline"} for call in data["calls"])
    assert get_metadata(result.function, MemoryMetadata) is not None
    cost = get_metadata(_calls(result.function)[-1], ComputeCostMetadata)
    cost_moved = get_metadata(_calls(result.function)[-1], TrafficMetadata)
    assert cost is not None
    asked = render_comment(cost_moved, opt_in=frozenset({"operands"}))
    assert " operands=0:r" in asked
    assert ",result:r" in asked
    assert "operands" not in render_comment(cost_moved)

    payload = json.loads(render_json(data))
    assert payload == data
    text = render_text(rendered)
    bound = payload["function_records"]["roofline"]
    assert f"# roofline ideal-ns={bound['ideal_ns']} bound-by={bound['bound_by']}" in text
    assert "# compute-cost flops=f32:256@256" in text
    assert "# traffic" not in text, "a dependency's own line was promoted into a bound"
    assert data["totals"]["traffic"] == {"gmem": {"read": 2048, "write": 1024}}
    assert "# peak-footprint" not in text
    assert "operands" not in text


def test_a_participant_is_charged_the_share_of_a_source_it_actually_reads() -> None:
    """A source nobody sharded is still read one tile at a time.

    A replicated value projects to the whole of itself, so charging that to
    every participant multiplied one read by the number of them: 132 CTAs were
    each billed the entire 67,584-byte source to fetch the 512 bytes they
    wanted. The Op's access relation is asked instead, in the same window.

    The broadcast operand keeps the rule honest the other way: one element read
    by every participant is one element, not the result's worth.
    """
    result, entry = _run(_replicated_source, "performance")
    calls = _calls(entry)
    loaded = next(
        call
        for call in calls
        if type(call.target).__name__ == "Reshard"
        and get_metadata(call, TrafficMetadata).at("gmem").read == 67_584
    )
    added = next(call for call in calls if type(call.target).__name__ == "Binary")

    cost_moved = get_metadata(loaded, TrafficMetadata)
    assert cost_moved.per_unit_at("gmem") == TrafficBytes(read=512)
    assert cost_moved.at("gmem") == TrafficBytes(read=67_584)
    assert cost_moved.per_unit_at("rmem") == TrafficBytes(write=512)


    broadcast_moved = get_metadata(added, TrafficMetadata)
    assert broadcast_moved.per_unit_at("rmem") == TrafficBytes(read=516, write=512)

    record = get_metadata(loaded, PerformanceMetadata)
    services = _replicated_source.resolve_target().get_facts(PerformanceServiceFacts)
    tile_ns = -(-(512 * 1_000_000_000) // services.bandwidth("gmem"))
    assert record.timeline.end_ns - record.timeline.start_ns == tile_ns
    del result


def test_a_tuple_result_moves_what_every_field_of_it_states() -> None:
    """Two outputs are two boundaries, and the result moved both of them.

    A tuple result has one boundary per field. Reading only the first drops the
    rest; reading the tuple as one value has no element count at all and falls
    back to the Type the relation replaced. The fields differ in element size,
    so the sum is not twice either: 4 values at 4 bytes and 4 indices at 8 come
    to 48, where the first field alone would say 16. The source is replicated,
    so the read side is one participant's share.
    """
    _result, entry = _run(_replicated_tuple, ("compute-cost", "memory"))

    chosen = next(call for call in _calls(entry) if type(call.target).__name__ == "TopK")
    cost_moved = get_metadata(chosen, TrafficMetadata)
    assert cost_moved.at("rmem") == TrafficBytes(read=67_584, write=6_336)
    assert cost_moved.per_unit_at("rmem") == TrafficBytes(read=512, write=48)


