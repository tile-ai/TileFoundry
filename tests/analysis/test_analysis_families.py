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
from tests.fixtures.placed.flash_split_k_decode import FlashSplitKDecode
from tests.fixtures.placed.moe_mega_kernel import MoEMegaKernel
from tests.fixtures.placed.prefill_decode_attention import PrefillDecodeAttention
from tests.fixtures.placed.square_cuda import Model as SquareCuda
from tilefoundry import func, module
from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryHierarchyFacts,
    MemoryMetadata,
    MemoryRelationKind,
    ParallelCapacityFacts,
    RooflineMetadata,
    ThroughputFacts,
    TimelineMetadata,
    TimelineSummaryMetadata,
)
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import _mesh_image, _result_placement, _timeline_placements
from tilefoundry.analysis.compute_cost import _call_movement
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.walk import postorder
from tilefoundry.dsl import ConstTensor, DimVar, Mesh, Tensor, Topology, tf
from tilefoundry.inspection.analysis_report import render_json, render_text, report
from tilefoundry.ir.core import Call, Constant, Tuple, Var, get_metadata
from tilefoundry.ir.hir import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.tensor.slice import Slice
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
class _UnmaterializedValueCosts:
    @func
    def main(source: Tensor[(8, 4), "f32"]):
        shape_value = tf.shape_of(source)[1]
        return tf.zeros(shape=(shape_value,), dtype="i64", storage="umat")


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _RuntimeCoordinateCosts:
    @func
    def main(source: Tensor[(8, 4), "f32"], start: Tensor[(), "i64"]):
        return source[start : start + 4, :]


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _UnmaterializedIndexTensorCosts:
    @func
    def main():
        row_positions = tf.reshape(tf.arange(128), new_shape=(1, 128))
        column_positions = tf.reshape(tf.arange(128), new_shape=(128, 1))
        return row_positions <= column_positions


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
class _WindowCost:
    @func
    def main(source: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
        out = tf.add(seed, seed)
        for row in tile(10, 4):
            out = tf.add(source[row, :], seed)
        return out


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


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 168),))
def _placed_wide_grid(source: Tensor[(1024,), "f32"]):
    with Mesh(("cta",), layout=(168,), names=("tile",)) as _cta:
        placed = tf.reshard(source, (1024,), "gmem")
        return tf.add(placed, placed)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def _reshard_boundary(source: Tensor[(1,), "f32"]):
    with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
        local = tf.reshard(source, (1 @ cta.tile,), "rmem")
        moved = tf.reshard(local, (1 @ cta.tile,), "rmem")
        return tf.add(moved, moved)


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def _zero_cost_view(source: Tensor[(1,), "f32"]):
    with Mesh(("cta",), layout=(1,), names=("tile",)) as cta:
        local = tf.reshard(source, (1 @ cta.tile,), "gmem")
        parts = tf.split(local, axis=0, num_splits=1)
        return parts[0]


@func(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 1),))
def _no_op_reshard(source: Tensor[(1,), "f32"]):
    return tf.reshard(source)


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
    cta_result = analyze(_thread_sharded, entry, analysis="compute-cost")

    record = get_metadata(_calls(cta_result.function)[-1], ComputeCostMetadata)
    assert record is not None
    assert cta_result.level == "cta"
    assert record.flops == (("f32", 8),)
    assert record.flops_per_unit == (("f32", 8),)
    traffic = record.traffic

    thread_result = analyze(_thread_sharded, entry, analysis="compute-cost", level="thread")
    record = get_metadata(_calls(thread_result.function)[-1], ComputeCostMetadata)
    assert record is not None
    assert thread_result.level == "thread"
    assert record.flops == (("f32", 8),)
    assert record.flops_per_unit == (("f32", 2),)
    assert record.traffic == traffic


def test_an_unsharded_call_reports_the_same_global_and_per_unit_work() -> None:
    cuda_entry = _CudaAdd.entry_function()
    amx_entry = _AmxAdd.entry_function()

    cuda = analyze(_CudaAdd, cuda_entry, analysis="compute-cost")
    amx = analyze(_AmxAdd, amx_entry, analysis="compute-cost")

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
        records.append((cost, bound))

    (
        (plain_cost, plain_bound),
        (split_cost, split_bound),
        (
            broadcast_cost,
            broadcast_bound,
        ),
    ) = records
    total_flops = plain_cost.flops[0]
    assert split_cost.flops == broadcast_cost.flops == (total_flops,)
    assert broadcast_cost.flops_per_unit == plain_cost.flops
    split_per_unit = split_cost.flops_per_unit[0]
    assert (split_per_unit[0], split_per_unit[1] * _MatmulLayouts.topologies[0].size) == (
        total_flops[0],
        total_flops[1],
    )
    assert split_cost.traffic == plain_cost.traffic == broadcast_cost.traffic
    assert split_bound == plain_bound == broadcast_bound


def test_function_call_carries_the_callee_per_unit_work() -> None:
    entry = _NestedSplitAdd.entry_function()
    result = analyze(_NestedSplitAdd, entry, analysis="compute-cost")

    record = get_metadata(_calls(result.function)[-1], ComputeCostMetadata)
    assert record is not None
    assert record.flops == (("f32", 256),)
    assert record.flops_per_unit == (("f32", 64),)


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
    rotated_result = analyze(_Rotated, rotated, analysis="compute-cost")
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
    allocated_result = analyze(_Allocated, allocated, analysis="compute-cost")
    zeros = get_metadata(
        next(
            call
            for call in _calls(allocated_result.function)
            if type(call.target).__name__ == "Zeros"
        ),
        ComputeCostMetadata,
    )
    assert zeros is not None
    assert zeros.flops == ()
    traffic = zeros.traffic_at("gmem")
    assert traffic.write == 64 * 4
    assert traffic.read == 0


def test_index_select_reads_the_rows_it_names_and_not_the_table() -> None:
    result, function = _run(_IndexSelected, "compute-cost")

    record = get_metadata(_calls(function)[-1], ComputeCostMetadata)
    assert record is not None
    traffic = record.traffic_at("gmem")
    rows = 4 * 64 * 4
    assert traffic.read == rows + 4 * 4
    assert traffic.write == rows

    payload = json.loads(render_json(report(result)))
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


def test_movement_costs_follow_each_operations_materialization() -> None:
    functions = {function.name: function for function in _MovementCosts.functions}

    for name in ("row", "column"):
        function = functions[name]
        result = analyze(_MovementCosts, function, analysis="memory")
        (move,) = _calls(result.function)
        record = get_metadata(move, ComputeCostMetadata)
        assert record is not None
        assert record.traffic_at("gmem").total_bytes == 0
        footprint = get_metadata(result.function, MemoryMetadata)
        assert footprint is not None
        assert all(item.persistent for item in footprint.lifetimes)

    materialized = functions["materialized"]
    result = analyze(_MovementCosts, materialized, analysis="memory")
    transpose, reshape = _calls(result.function)
    transpose_cost = get_metadata(transpose, ComputeCostMetadata)
    reshape_cost = get_metadata(reshape, ComputeCostMetadata)
    assert transpose_cost is not None
    moved = 1024 * 2048 * 4
    assert transpose_cost.traffic_at("gmem") == TrafficBytes(read=moved, write=moved)
    assert reshape_cost is not None
    assert reshape_cost.traffic_at("gmem").total_bytes == 0
    footprint = get_metadata(result.function, MemoryMetadata)
    assert footprint is not None
    assert any(not item.persistent and item.bytes == moved for item in footprint.lifetimes)

    copied = functions["copied"]
    result = analyze(_MovementCosts, copied, analysis="compute-cost")
    selected, concat = _calls(result.function)
    selected_cost = get_metadata(selected, ComputeCostMetadata)
    concat_cost = get_metadata(concat, ComputeCostMetadata)
    assert selected_cost is not None
    selected_bytes = 256 * 2048 * 4
    assert selected_cost.traffic_at("gmem").total_bytes == 0
    assert concat_cost is not None
    assert concat_cost.traffic_at("gmem") == TrafficBytes(
        read=2 * selected_bytes,
        write=2 * selected_bytes,
    )


def test_reshard_costs_no_op_and_cross_storage_copy() -> None:
    no_op, no_op_function = _run(_no_op_reshard, "compute-cost")
    assert no_op.executed == ("compute-cost",)
    no_op_cost = get_metadata(_calls(no_op_function)[0], ComputeCostMetadata)
    assert no_op_cost is not None
    assert no_op_cost.traffic == ()
    assert no_op_cost.operands == (TrafficBytes(), TrafficBytes())

    copied = analyze(SquareCuda, SquareCuda.entry_function(), analysis="compute-cost")
    copied_in = _calls(copied.function)[0]
    copied_cost = get_metadata(copied_in, ComputeCostMetadata)
    assert copied_cost is not None
    moved = 168 * 4
    assert copied_cost.traffic == (
        ("gmem", TrafficBytes(read=moved)),
        ("rmem", TrafficBytes(write=moved)),
    )
    assert copied_cost.operands == (
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
    shape_value = zeros.target.shape[0]
    assert shape_value.type.storage is StorageKind.UMAT
    assert shape_value not in zeros.args
    assert zeros.type.storage is StorageKind.UMAT
    traffic, operands = _call_movement(zeros, Cost({}, (TrafficBytes(write=16),)))
    assert operands == (TrafficBytes(write=16),)
    assert traffic == ()


def test_mixed_runtime_coordinates_are_charged_per_leaf() -> None:
    function = _RuntimeCoordinateCosts.entry_function()
    result = analyze(_RuntimeCoordinateCosts, function, analysis="compute-cost")

    (slice_call,) = _calls(result.function)
    starts = slice_call.args[1]
    assert tuple(str(field.storage) for field in starts.type.fields) == (
        "gmem",
        "umat",
    )
    record = get_metadata(slice_call, ComputeCostMetadata)
    assert record is not None
    assert record.traffic == (
        ("gmem", TrafficBytes(read=8)),
        ("rmem", TrafficBytes(read=8)),
    )
    assert record.operands[1] == TrafficBytes(read=16)


def test_non_scalar_unmaterialized_operands_are_charged_at_rmem() -> None:
    function = _UnmaterializedIndexTensorCosts.entry_function()
    result = analyze(_UnmaterializedIndexTensorCosts, function, analysis="compute-cost")

    (binary,) = [call for call in _calls(result.function) if isinstance(call.target, Binary)]
    assert tuple(arg.type.shape for arg in binary.args) == ((1, 128), (128, 1))
    assert all(arg.type.storage is StorageKind.UMAT for arg in binary.args)
    record = get_metadata(binary, ComputeCostMetadata)
    assert record is not None
    assert record.traffic == (("rmem", TrafficBytes(read=2_048)),)
    assert record.operands == (
        TrafficBytes(read=1_024),
        TrafficBytes(read=1_024),
        TrafficBytes(write=2_048),
    )


def test_non_divisible_window_cost_is_a_full_tile_upper_bound() -> None:
    function = _WindowCost.entry_function()
    result = analyze(_WindowCost, function, analysis="compute-cost")
    adds = [call for call in _calls(result.function) if isinstance(call.target, Binary)]
    loop_cost = get_metadata(adds[-1], ComputeCostMetadata)
    root_cost = get_metadata(result.function, ComputeCostMetadata)

    assert loop_cost is not None
    assert loop_cost.flops == (("f32", 4 * 4),)
    assert root_cost is not None
    assert root_cost.flops == (("f32", 4 * 4 + 3 * 4 * 4),)


def test_a_sharded_shared_tile_fits_once_and_advises_on_its_peak() -> None:
    """The H200 capacity is an input fact; the peak is two live tile lifetimes."""
    functions = {function.name: function for function in _SharedTile.functions}
    split = functions["split"]
    result = analyze(_SharedTile, split, analysis="memory")

    record = get_metadata(result.function, MemoryMetadata)
    assert record is not None
    smem = next(item for item in record.footprint if item.level == "smem")
    assert smem.capacity_bytes == 232_448
    largest_tile = max(item.bytes for item in record.lifetimes if item.level == "smem")
    assert smem.peak_bytes == 2 * largest_tile
    assert any(
        f"smem peak is {smem.peak_bytes} B" in note
        and "order-dependent" in note
        and "not a bound" in note
        for note in record.advisories
    )

    with pytest.raises(AnalysisError, match=r"27878400 B in smem"):
        analyze(_SharedTile, functions["broadcast"], analysis="memory")


def test_memory_footprints_follow_the_owner_recorded_by_the_target() -> None:
    matmul = next(fn for fn in _MatmulLayouts.functions if fn.name == "split")
    result = analyze(_MatmulLayouts, matmul, analysis="memory")
    gmem = get_metadata(result.function, MemoryMetadata)
    assert gmem is not None
    gmem_lifetimes = {item.binding: item.bytes for item in gmem.lifetimes if item.level == "gmem"}
    assert gmem_lifetimes["v0"] == gmem_lifetimes["lhs"]

    shared = next(fn for fn in _SharedTile.functions if fn.name == "split")
    result = analyze(_SharedTile, shared, analysis="memory")
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
            split_result = result
        call = _calls(result.function)[-1]
        matmul_records.append(
            (
                get_metadata(call, ComputeCostMetadata),
                get_metadata(call, RooflineMetadata),
            )
        )
    plain_cost, plain_bound = matmul_records[0]
    split_cost, _split_bound = matmul_records[1]

    shared = next(function for function in _SharedTile.functions if function.name == "split")
    shared_result = analyze(_SharedTile, shared, analysis="memory")
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
        get_metadata(expr, ComputeCostMetadata).traffic
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
        "placed_ideal_ns": (3_496, 65_447),
        "placed_traffic": (
            {
                "gmem": {"read": 8_390_656, "write": 8_388_608},
                "rmem": {"read": 17_256, "write": 0},
                    "smem": {"read": 21_669_568, "write": 21_432_000},
            },
            {
                "gmem": {"read": 4_194_304, "write": 8_388_608},
                "rmem": {"read": 279_488, "write": 0},
                    "smem": {"read": 794_492_928, "write": 484_114_432},
            },
        ),
        "flash_split_traffic": {
            "gmem": {"read": 2_132_480, "write": 35_328},
            "rmem": {"read": 400, "write": 104},
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


def test_timeline_refuses_unplaced_results_before_dependencies_run(monkeypatch) -> None:
    """A topology declaration does not place a value or admit dependency writes."""
    compute = analyze(_wide_grid, _entry(_wide_grid), analysis="compute-cost")
    roofline = analyze(_wide_grid, _entry(_wide_grid), analysis="roofline")
    assert get_metadata(compute.function, ComputeCostMetadata) is not None
    assert get_metadata(roofline.function, RooflineMetadata) is not None

    placed, function = _run(_placed_wide_grid, "timeline")
    assert placed.executed == ("compute-cost", "timeline")
    assert all(get_metadata(call, TimelineMetadata) is not None for call in _calls(function))

    ran = False

    def unexpected(*_args, **_kwargs):
        nonlocal ran
        ran = True

    monkeypatch.setattr(compute_cost, "analyze_compute_cost", unexpected)
    with pytest.raises(
        AnalysisError,
        match=r"test_analysis_families.py:.*has no cta placement",
    ):
        _run(_wide_grid, "timeline")
    assert not ran


def test_mega_kernel_preserves_placement_costs_and_timeline_order() -> None:
    """The expanded placed program keeps its slices, costs, and dependency order."""
    result = analyze(
        MoEMegaKernel,
        MoEMegaKernel.entry_function(),
        analysis=("compute-cost", "memory", "roofline", "timeline"),
    )
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

    placements = _timeline_placements(
        MoEMegaKernel,
        result.function,
        "cta",
        MoEMegaKernel.resolve_target().get_facts(ThroughputFacts),
    )
    prepared = set(placements.values())

    assert frozenset(range(120)) in prepared
    assert frozenset(range(120, 132)) in prepared
    assert frozenset(range(132)) in prepared

    views = [call for call in calls if type(call.target).__name__ == "Reshard"]
    assert len(views) == 4
    for view in views:
        view_cost = get_metadata(view, ComputeCostMetadata)
        assert view_cost is not None
        assert view_cost.traffic == ()
        assert view_cost.traffic_per_unit == ()
        assert view_cost.operands == (TrafficBytes(), TrafficBytes())

    call_costs = [get_metadata(call, ComputeCostMetadata) for call in calls]
    assert all(call_cost is not None for call_cost in call_costs)
    summed = TrafficBytes(
        read=sum(call_cost.traffic_at("gmem").read for call_cost in call_costs),
        write=sum(call_cost.traffic_at("gmem").write for call_cost in call_costs),
    )
    root_cost = get_metadata(result.function, ComputeCostMetadata)
    memory = get_metadata(result.function, MemoryMetadata)
    roofline = get_metadata(result.function, RooflineMetadata)
    assert root_cost is not None
    assert memory is not None
    assert roofline is not None
    assert dict(root_cost.flops) == {"f32": 23_040}
    assert dict(root_cost.flops) == {
        "f32": sum(dict(call_cost.flops).get("f32", 0) for call_cost in call_costs)
    }
    assert root_cost.traffic_at("gmem") == summed == TrafficBytes(
        read=122_880,
        write=92_160,
    )
    assert memory.traffic == (("gmem", summed),)
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
    assert cost is not None
    assert cost.traffic == (("gmem", TrafficBytes(read=30_720, write=30_720)),)
    assert cost.traffic_per_unit == (("gmem", TrafficBytes(read=256, write=256)),)
    assert cost.traffic_per_unit_at("gmem") == TrafficBytes(read=256, write=256)
    assert (
        "bytes=gmem:r30720/w30720 per-unit-bytes=gmem:r256/w256 operands="
        in cost.format_comment()
    )

    records = tuple(get_metadata(call, TimelineMetadata) for call in calls)
    assert all(record is not None for record in records)

    routed_in, routed, routed_out, shared_in, shared, shared_out, join = records
    assert routed_in is not None and routed is not None and routed_out is not None
    assert shared_in is not None and shared is not None and shared_out is not None
    assert join is not None

    assert [
        (record.start_ns, record.end_ns)
        for record in (
            routed_in,
            routed,
            routed_out,
            shared_in,
            shared,
            shared_out,
            join,
        )
    ] == [(0, 0), (0, 15), (15, 15), (0, 0), (0, 141), (141, 141), (141, 2_676)]
    assert (routed.end_ns - routed.start_ns, shared.end_ns - shared.start_ns) == (
        15,
        141,
    )
    assert routed_in.start_ns == shared_in.start_ns == 0

    routed_positions = placements[id(calls[1])]
    shared_positions = placements[id(calls[4])]
    assert routed_positions.isdisjoint(shared_positions)
    assert routed.start_ns < shared.end_ns and shared.start_ns < routed.end_ns

    whole_positions = placements[id(calls[2])]
    assert whole_positions == placements[id(calls[5])] == frozenset(range(132))
    assert (
        routed_out.end_ns <= shared_out.start_ns
        or shared_out.end_ns <= routed_out.start_ns
    )
    assert whole_positions != shared_positions
    assert whole_positions & shared_positions
    assert join.start_ns >= max(routed_out.end_ns, shared_out.end_ns)

    summary = get_metadata(result.function, TimelineSummaryMetadata)
    assert summary == TimelineSummaryMetadata(
        local_makespan_ns=2_676,
        waves=1,
        estimated_kernel_ns=2_676,
    )


def test_timeline_scales_one_local_schedule_by_root_capacity() -> None:
    """Root geometry and capacity scale waves, never occurrence intervals."""
    results = [
        analyze(
            _ScalableGrid,
            _ScalableGrid.entry_function(),
            analysis="timeline",
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
            analysis="timeline",
            dims={"grid": 265},
        )
    )

    intervals = [
        tuple(get_metadata(call, TimelineMetadata) for call in _calls(result.function))
        for result in results
    ]
    summaries = [
        get_metadata(result.function, TimelineSummaryMetadata) for result in results
    ]
    assert intervals[0] == intervals[1] == intervals[2]
    assert all(summary is not None for summary in summaries)
    first, wider, constrained_summary = summaries
    assert first is not None and wider is not None and constrained_summary is not None
    assert {
        first.local_makespan_ns,
        wider.local_makespan_ns,
        constrained_summary.local_makespan_ns,
    } == {15}
    assert (first.waves, wider.waves, constrained_summary.waves) == (1, 3, 5)
    assert (
        first.estimated_kernel_ns,
        wider.estimated_kernel_ns,
        constrained_summary.estimated_kernel_ns,
    ) == (15, 45, 75)

    data = report(results[1])
    assert set(data["function_records"]["timeline"]) == {
        "local_makespan_ns",
        "waves",
        "estimated_kernel_ns",
    }
    assert all(
        set(row["timeline"]) == {"start_ns", "end_ns", "trips", "stride_ns"}
        for row in data["calls"]
    )


def test_placement_projection_rejects_the_wrong_level_and_invalid_images() -> None:
    with pytest.raises(
        AnalysisError,
        match=r"selected topology level 'cta'.*placed at level.s. 'thread'",
    ):
        analyze(_thread_sharded, _entry(_thread_sharded), analysis="timeline")

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


def test_timeline_refuses_traffic_only_at_an_unmodelled_storage_level() -> None:
    with pytest.raises(
        AnalysisError,
        match=r"traffic is only at unmodelled storage level.*'rmem'.*'gmem'",
    ):
        _run(_reshard_boundary, "timeline")


def test_a_zero_cost_structural_occurrence_has_an_empty_timeline_interval() -> None:
    result, entry = _run(_zero_cost_view, "timeline")

    view = next(
        call for call in _calls(entry) if type(call.target).__name__ == "TupleGetItem"
    )
    record = get_metadata(view, TimelineMetadata)
    assert record is not None
    assert record.start_ns == record.end_ns
    timelines = [
        row["timeline"] for row in report(result)["calls"] if "timeline" in row
    ]
    assert all(
        set(item)
        == {"start_ns", "end_ns", "trips", "stride_ns"}
        for item in timelines
    )
    assert any(
        item["start_ns"] == item["end_ns"]
        and item["trips"] == 1
        and item["stride_ns"] == 0
        for item in timelines
    )


def test_an_unplaced_structural_occurrence_ignores_unmodelled_traffic() -> None:
    _result, entry = _run(_unplaced_structural, "timeline")

    (view,) = _calls(entry)
    cost = get_metadata(view, ComputeCostMetadata)
    timeline = get_metadata(view, TimelineMetadata)
    assert cost is not None and timeline is not None
    assert cost.flops_per_unit == ()
    assert cost.traffic_per_unit_at("gmem").total_bytes == 0
    assert cost.traffic_per_unit_at("rmem") == TrafficBytes(read=8, write=8)
    assert (timeline.start_ns, timeline.end_ns) == (0, 0)


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

    data = report(result)

    assert data["requested"] == ["roofline"]
    assert data["executed"] == ["compute-cost", "memory", "roofline"]
    assert set(data["function_records"]) == {"memory", "roofline"}
    memory = data["function_records"]["memory"]
    assert set(memory) == {"footprint"}
    assert all(set(item) == {"level", "peak_bytes"} for item in memory["footprint"])
    assert data["totals"]["flops"] == {"f32": 256}
    assert all(set(call) == {"value", "roofline"} for call in data["calls"])
    assert get_metadata(result.function, MemoryMetadata) is not None
    cost = get_metadata(_calls(result.function)[-1], ComputeCostMetadata)
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
