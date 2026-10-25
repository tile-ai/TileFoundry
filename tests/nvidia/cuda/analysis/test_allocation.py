"""Physical placement, semantic aliases, and call-boundary proofs."""

from pathlib import Path

import pytest

from tilefoundry import func, module
from tilefoundry.analysis import (
    MemoryMetadata,
    RegionMemoryMetadata,
    analyze,
)
from tilefoundry.analysis.access import AliasKind
from tilefoundry.analysis.allocation import AllocationSummaryMetadata
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.liveness import LivenessMetadata
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.schedule import finalize
from tilefoundry.visitor_registry.verify import verify_module
from tilefoundry_nvidia import CudaTarget


def test_insert_slice_keeps_its_address_through_sequential_carry_loops():
    path = Path(__file__).parents[1] / "fixtures/logical/sequential_carries.py"
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    inserts = [
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call) and isinstance(value.target, InsertSlice)
    ]
    assert len(inserts) == 3
    values = {
        value.value: value
        for value in get_metadata(result.function, AllocationSummaryMetadata).values
    }
    for insert in inserts:
        memory = get_metadata(insert, MemoryMetadata)
        link = next(
            alias
            for alias in values[SSAValue(insert)].aliases
            if alias.kind is AliasKind.INSERT_SLICE
        )
        assert link.relation.relation.is_identity()
        assert link.target.expr is insert.args[0]
        assert memory.buffer_bytes is None and memory.offsets == ()
    verify_module(finalize(result, authored=module.entry_function()))


@pytest.mark.parametrize("later_use", [False, True])
def test_retained_elementwise_return_overwrites_only_a_dead_actual(later_use):
    path = Path(__file__).parents[1] / f"fixtures/logical/fresh_return_{int(later_use)}.py"
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    call = next(
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call) and isinstance(value.target, Function)
    )
    actual = get_metadata(call.args[0], MemoryMetadata)
    returned = get_metadata(call, MemoryMetadata)
    assert actual.offsets and returned.offsets
    assert (returned.offsets == actual.offsets) is (not later_use)
    assert returned.buffer_bytes == actual.buffer_bytes == 32
    summary = get_metadata(result.function, AllocationSummaryMetadata)
    values = {value.value: value for value in summary.values}
    assert (
        AllocationSummaryMetadata(tuple(values.values())).root(values[SSAValue(call)]) is AllocationSummaryMetadata(tuple(values.values())).root(values[SSAValue(call.args[0])])
    ) is (not later_use)


def test_dynamic_selection_stays_resident() -> None:
    """A captured dynamic view pins both possible leaves without enabling reuse."""

    @module(
        entry="choose",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1),),
    )
    class DynamicSelection:
        @func
        def choose(k: Tensor[(), "i32"]):
            with Mesh(("cta",), (1,), names=("block",)) as cta:
                first = tf.zeros(Tensor[(4,), "bf16", (4,), "smem"])
                second = tf.zeros(Tensor[(4,), "bf16", (4,), "smem"])
                choices = (first, second)
                selected = choices[k]
                with cta[:] as _inner:
                    picked = selected
                result = tf.add(picked, picked)
                return result

    result = analyze(
        *check_program(DynamicSelection, DynamicSelection.entry_function()),
        analysis="memory",
    )
    selected = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, TupleGetItem)
    )
    first, second = selected.args[0].elements
    consumer = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and len(expr.args) == 2
        and all(isinstance(arg, MeshRegion) for arg in expr.args)
    )
    records = [get_metadata(value, MemoryMetadata) for value in (first, second, consumer)]
    assert all(record.buffer_bytes == 8 and len(record.offsets) == 1 for record in records)
    spans = sorted(
        (record.offsets[0], record.offsets[0] + record.buffer_bytes) for record in records
    )
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
    moved = get_metadata(result.function, RegionMemoryMetadata)
    assert moved.peak_for("smem").peak_bytes == 40
    lifetimes = {life.value: life for life in moved.lifetimes}
    output = lifetimes[SSAValue(consumer)]
    assert all(
        lifetimes[SSAValue(leaf)].last_used_at < output.defined_at for leaf in (first, second)
    )
    assert get_metadata(selected, MemoryMetadata).buffer_bytes is None
    assert not get_metadata(selected, MemoryMetadata).offsets
    summary = get_metadata(result.function, AllocationSummaryMetadata)
    by_value = {value.value: value for value in summary.values}
    assert not by_value[SSAValue(consumer)].aliases
    assert SSAValue(selected) not in by_value
    liveness = get_metadata(result.function, LivenessMetadata)
    for index in range(len(selected.args[0].elements)):
        assert (
            liveness.interval_of(SSAValue(selected.args[0], index)).last_used_at
            == liveness.timeline_end
        )
    for region in consumer.args:
        assert by_value.get(SSAValue(region)) is None
        assert all(by_value.get(SSAValue(param)) is None for param in region.params)
    allocated = by_value[SSAValue(consumer)]
    assert all(
        summary.root(allocated) is not summary.root(SSAValue(leaf))
        for leaf in (first, second)
    )
