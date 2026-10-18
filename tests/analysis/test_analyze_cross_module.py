"""Analyze a root that reaches a child Module as one kernel.

A call into a child is device work in the current occurrence, so its cost lands
in the caller's totals and repeating the site or looping over it multiplies that
work rather than adding an occurrence. What the fixed-dimension query has to
establish first is that the two ends share one execution context.

No GPU, no codegen, no runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import ClassVar

import isl
import pytest

from tests.fixtures.logical.hir_composition import REFERENCE_PROGRAMS, CrossModule, Expert
from tests.fixtures.placed.moe_mega_kernel import MoEMegaKernel
from tests.fixtures.shapes.scaled_modules import FusedScaledParent, ScaledChild, StackedScaledParent
from tests.models.qwen3_1_7b.case import CASE as QWEN_CASE
from tests.models.qwen3_1_7b.model import Qwen3_1_7B
from tilefoundry.analysis.access import SSAValue
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.footprint import ReachedAddresses, resolve_footprint
from tilefoundry.analysis.metadata import (
    BindingValuation,
    ComputeCostMetadata,
    Footprint,
    LoopNestValuation,
    MemoryMetadata,
    RegionMemoryMetadata,
)
from tilefoundry.analysis.precision import AnalysisPrecision
from tilefoundry.analysis.registry import Analyzer
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, Constant, Tuple, Var, attach_metadata, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.tensor.empty import Empty
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.types import DType, Layout, Mesh, TensorType, TupleType
from tilefoundry.ir.types.layout import ComposedLayout, flatten
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.ir.types.utils import tensor_types
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget, Target
from tilefoundry.visitor_registry.access_relation import AccessRelation
from tilefoundry.visitor_registry.call_frame import CallFrame, CallFrameVisitor

_H200 = "nvidia.h200_sxm"
_CTA = (Topology("cta", 132),)


def _matmul_records(result) -> tuple[tuple[ComputeCostMetadata, MemoryMetadata, RegionMemoryMetadata], ...]:
    """Both halves of each occurrence's MatMul record."""
    records = []
    def visit(frame):
        for expr in collect_exprs(frame.function.body):
            if not isinstance(expr, Call):
                continue
            child = CallFrameVisitor.callee_frame(expr, frame)
            if child is not None:
                visit(child)
            elif isinstance(expr.target, MatMul):
                record = get_metadata(expr, ComputeCostMetadata)
                moved = get_metadata(expr, MemoryMetadata)
                assert record is not None
                records.append((record, moved))

    visit(CallFrame(result.module, result.function))
    return tuple(records)


def _flops(records) -> dict[str, int]:
    total: dict[str, int] = {}
    for item in records:
        record = item[0] if isinstance(item, tuple) else item
        for name, spread in record.flops.kinds:
            value = spread.total
            total[name] = total.get(name, 0) + value
    return total


def _traffic(records) -> dict[str, int]:
    total: dict[str, int] = {}
    for _record, moved in records:
        assert moved is not None, "traffic was asked of a run that did not measure it"
        for name, spread in moved.traffic.storage.kinds:
            total[name] = total.get(name, 0) + spread.total.total_bytes
    return total


def test_a_repeated_call_site_counts_its_work_again() -> None:
    @module(entry="twice", target=CudaTarget(_H200), topologies=_CTA)
    class _Twice:
        mm = Expert

        @func
        def twice(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return tf.add(mm(x), mm(x))  # noqa: F821

    one_result = analyze(
        *check_program(CrossModule, CrossModule.entry_function()),
        analysis="compute-cost",
    )
    two_result = analyze(*check_program(_Twice, _Twice.entry_function()), analysis="compute-cost")
    one_records = _matmul_records(one_result)
    two_records = _matmul_records(two_result)
    one = _flops(one_records)
    two = _flops(two_records)

    assert len(one_records) == 1 and two_records == one_records * 2
    assert one and two == {name: 2 * value for name, value in one.items()}


def test_a_child_call_a_loop_varies_is_counted_once_per_trip() -> None:
    @module(entry="looped", target=CudaTarget(_H200), topologies=_CTA)
    class _Looped:
        mm = Expert

        @func
        def looped(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            acc = x
            for _ in range(3):
                acc = mm(acc)  # noqa: F821
            return acc

    one_result = analyze(
        *check_program(CrossModule, CrossModule.entry_function()),
        analysis="compute-cost",
    )
    looped_result = analyze(
        *check_program(_Looped, _Looped.entry_function()),
        analysis="compute-cost",
    )
    one_occurrence = _matmul_records(one_result)
    looped_occurrence = _matmul_records(looped_result)
    one_root = get_metadata(one_result.function, ComputeCostMetadata)
    looped_root = get_metadata(looped_result.function, ComputeCostMetadata)

    assert len(one_occurrence) == 1 and looped_occurrence == one_occurrence
    assert one_root is not None and looped_root is not None
    assert _flops((looped_root,)) == {
        name: 3 * value for name, value in _flops((one_root,)).items()
    }


def test_the_weight_traffic_of_a_fused_root_is_what_its_callees_read() -> None:
    @module(entry="direct", target=CudaTarget(_H200), topologies=_CTA)
    class _Direct:
        @func
        def direct(
            x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 8), "f32"]
        ) -> Tensor[(4, 8), "f32"]:
            return tf.matmul(x, w)

    direct_result = analyze(
        *check_program(_Direct, _Direct.entry_function()),
        analysis=("compute-cost", "memory"),
    )
    fused_result = analyze(
        *check_program(CrossModule, CrossModule.entry_function()),
        analysis=("compute-cost", "memory"),
    )
    direct = _traffic(_matmul_records(direct_result))
    fused = _traffic(_matmul_records(fused_result))

    assert direct and fused == direct


def test_a_reached_child_resolving_another_hierarchy_is_invalid() -> None:
    @module(entry="run", topologies=(Topology("warp", 4),))
    class _Warped:
        @func
        def run(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return tf.add(x, x)

    @module(entry="fused", target=CudaTarget(_H200), topologies=_CTA)
    class _Mismatch:
        warped = _Warped

        @func
        def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return warped(x)  # noqa: F821

    with pytest.raises(AnalysisError, match="different topology hierarchy") as caught:
        analyze(*check_program(_Mismatch, _Mismatch.entry_function()), analysis="compute-cost")
    assert "launch" not in str(caught.value)


def test_a_child_declaring_the_caller_hierarchy_is_accepted() -> None:
    extent = DimVar("child_topology_extent", 1, 132)
    hierarchy = (Topology("cta", extent),)

    @module(entry="run", topologies=hierarchy)
    class _Declared:
        @func
        def run(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return tf.add(x, x)

    @module(entry="fused", target=CudaTarget(_H200), topologies=hierarchy)
    class _Agreeing:
        declared = _Declared

        @func
        def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return declared(x)  # noqa: F821

    result = analyze(
        *check_program(_Agreeing, _Agreeing.entry_function(), dims={"child_topology_extent": 132}),
        analysis=("compute-cost", "memory"),
    )
    assert set(result.metadata_types) >= {ComputeCostMetadata, MemoryMetadata, RegionMemoryMetadata}


@pytest.mark.parametrize("name,root", REFERENCE_PROGRAMS, ids=[n for n, _ in REFERENCE_PROGRAMS])
def test_each_reference_program_is_one_analysable_kernel(name, root) -> None:
    """The shared programs measure as one root each; what a call means is not restated."""
    result = analyze(
        *check_program(root, root.entry_function()),
        analysis=("compute-cost", "memory"),
    )

    assert set(result.metadata_types) >= {ComputeCostMetadata, MemoryMetadata, RegionMemoryMetadata}


def _placed_primitives(fn) -> list[tuple[str, object]]:
    """Each costed primitive result in *fn* that a sliced Mesh reached."""
    found: list[tuple[str, object]] = []
    for expr in collect_exprs(fn.body):
        if not isinstance(expr, Call) or isinstance(expr.target, Function):
            continue
        for leaf in tensor_types(expr.type):
            if isinstance(leaf.layout, ShardLayout) and isinstance(
                leaf.layout.mesh.layout, ComposedLayout
            ):
                found.append((type(expr.target).__name__, leaf.layout.mesh.layout))
    return found


def test_each_placed_branch_keeps_its_slice_on_its_primitive_results() -> None:
    """A lexical Mesh scope places nothing; the reshard into it is what does.

    So the branch is read from what its results retained: the sub-box the slice
    selected, as its own extent and origin rather than an extent alone.
    """
    expected = {"routed_expert": ((120,), 0), "shared_expert": ((12,), 120)}
    reached = {
        fn.name: _placed_primitives(fn)
        for frame in list_call_paths(check_program(MoEMegaKernel, MoEMegaKernel.entry_function())[0]).values()
        for fn in (frame.function,)
    }

    for branch, (shape, offset) in expected.items():
        placed = reached[branch]
        assert placed, branch
        assert {op for op, _ in placed} - {"Reshard"}, branch
        for _op, layout in placed:
            assert flatten(layout.outer).shape == shape
            assert layout.offset == offset


def _stacked_program(count, trips=None):
    trips = count if trips is None else trips
    @module(entry="run", target=CudaTarget(_H200), topologies=_CTA)
    class Stack:
        layers = ModuleList(ScaledChild, count)

        @func
        def run(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            for i in range(trips):
                x = layers[i](x)  # noqa: F821 -- class-body binding
            return x

    return Stack


def _call_program(definition, *, count=1, indexed=False, name="run"):
    slot = ModuleList(definition, count) if indexed else ModuleList(definition)
    param = replace(definition.entry_function().params[0], is_const=False)
    calls = []
    value = param
    for index in range(count):
        target = slot
        if indexed:
            target = Call(TupleGetItem(), (slot, Constant(index, type=TensorType.scalar(DType.i32))),
                          type=slot.type.fields[index])
        value = Call(target, (value,), type=param.type)
        calls.append(value)
    if name == "twice":
        calls[1].args = (param,)
        value = Call(Binary(kind=BinaryKind.ADD), tuple(calls), type=param.type)
    if name == "looped":
        carry = replace(param, name="carry")
        iteration = Call(slot, (carry,), type=param.type)
        body = Tuple((iteration,), type=TupleType((param.type,)))
        loop = LoopRegion(
            induction_var=Var("i", type=TensorType.scalar(DType.i32)),
            params=(carry,), args=(param,), body=body, yield_values=(iteration,),
            start=0, extent=3, step=1, type=body.type,
        )
        value = loop
    function = Function.build(name=name, params=(param,), body=value, return_type=param.type)
    return Module(name, (function,), name, children={"layers" if indexed else "mm": slot},
                  target=CudaTarget(_H200), topologies=_CTA)


@pytest.mark.parametrize(
    "kind,flops,traffic,peak,persistent,occurrences",
    [("cross", 512, 512, 512, 384, 1), ("twice", 1056, 1408, 640, 384, 2),
     ("looped", 1536, 1536, 640, 384, 1),
     ("one", 4, 48, 48, 32, 1), ("unrolled", 12, 144, 80, 64, 3),
     ("stacked", 12, 144, 80, 64, 1)],
)
def test_retained_calls_match_the_memory_and_work_acceptance_values(
    kind, flops, traffic, peak, persistent, occurrences
):
    roots = {
        "cross": CrossModule,
        "twice": _call_program(Expert, count=2, name="twice"),
        "looped": _call_program(Expert, name="looped"),
        "one": replace(FusedScaledParent, topologies=_CTA),
        "unrolled": _call_program(ScaledChild, count=3, indexed=True),
        "stacked": replace(StackedScaledParent, topologies=_CTA),
    }
    root = roots[kind]
    result = analyze(
        *check_program(root, root.entry_function()),
        analysis=("compute-cost", "memory"),
    )
    cost = get_metadata(result.function, ComputeCostMetadata)
    moved = get_metadata(result.function, RegionMemoryMetadata)
    region = moved.peak_for("gmem")
    assert cost.flops.of("f32").total == flops
    assert moved.traffic.storage.of("gmem").total.total_bytes == traffic
    assert (region.peak_bytes, region.persistent_bytes) == (peak, persistent)
    assert len(result.functions) == 2
    frames = tuple(CallFrameVisitor.walk_frames(CallFrame(result.module, result.function)))
    sites = [
        (call, child)
        for frame in frames
        for call in collect_exprs(frame.function.body)
        if isinstance(call, Call)
        and (child := CallFrameVisitor.callee_frame(call, frame)) is not None
    ]
    assert len(sites) == occurrences
    if kind == "stacked":
        assert sum(isinstance(node, LoopRegion) for node in collect_exprs(result.function.body)) == 1
    for call, child in sites:
        assert get_metadata(child.function.body, ComputeCostMetadata) is not None
        assert get_metadata(call, ComputeCostMetadata) == get_metadata(child.function, ComputeCostMetadata)


def test_derived_nodes_and_bodies_do_not_grow_with_indexed_instance_count():
    sizes = []
    for count in (3, 28):
        root = _stacked_program(count)
        checked, function = check_program(root, root.entry_function())
        functions = tuple(frame.function for frame in list_call_paths(checked).values())
        sizes.append((sum(len(collect_exprs(body.body)) for body in functions),
                      len(functions)))
        assert sum(isinstance(node, LoopRegion) for node in collect_exprs(function.body)) == 1
        with pytest.raises(AnalysisError, match="derived body nodes.*node budget"):
            check_program(root, root.entry_function(), budget=sizes[-1][0] - 1)
    assert sizes[0] == sizes[1]
    assert sizes[0][1] == 2


def test_the_published_qwen_stack_shares_its_layer_definition_and_fits_the_budget():
    elements = Qwen3_1_7B.children["layers"].elements
    assert len(elements) == 28
    assert all(element is elements[0] for element in elements)
    root = QWEN_CASE.build()
    checked, _function = check_program(root, root.lookup("decoder_layer"), dims={"ctx_len": 128})
    assert sum(len(collect_exprs(frame.function.body)) for frame in list_call_paths(checked).values()) <= 10_000


def _direct_program(callee, params, calls):
    body = Tuple(tuple(calls), type=TupleType(tuple(call.type for call in calls)))
    function = Function.build(name="root", params=params, body=body, return_type=body.type)
    return Module("DirectCalls", (callee, function), "root", target=CudaTarget(_H200), topologies=_CTA)


@pytest.mark.parametrize("difference", ["layout", "storage", "mesh"])
def test_one_definition_rejects_different_call_signatures(difference):
    type_ = TensorType((4, 8), DType.f32, None, StorageKind.UMAT)
    left = Var("left", type=type_)
    changed = {
        "layout": replace(type_, layout=Layout((4, 8), (8, 1))),
        "storage": replace(type_, storage=StorageKind.SMEM),
        "mesh": type_,
    }[difference]
    right = Var("right", type=changed)
    held = Var("x", type=type_)
    callee = Function.build(name="helper", params=(held,), body=held, return_type=type_)
    first = Call(callee, (left,), type=type_)
    second = Call(callee, (right,), type=changed)
    if difference == "mesh":
        first = MeshRegion(Mesh(_CTA, Layout((1,), (1,)), ("first",)), first, type=type_)
        second = MeshRegion(Mesh(_CTA, Layout((1,), (1,)), ("second",)), second, type=changed)
    root = _direct_program(callee, (left, right), (first, second))
    with pytest.raises(AnalysisError, match="callee 'helper'.*root/v0.*root/v1.*versus"):
        check_program(root, root.entry_function())


def test_recursive_call_paths_and_missing_callee_bodies_are_diagnosed():
    type_ = TensorType((4,), DType.f32, None, StorageKind.UMAT)
    param = Var("x", type=type_)
    callee = Function.build(name="helper", params=(param,), body=None, return_type=type_)
    root = _direct_program(callee, (param,), (Call(callee, (param,), type=type_),))
    with pytest.raises(AnalysisError, match="cannot analyze Function 'helper' without a body"):
        check_program(root, root.entry_function())
    callee.body = Call(callee, (param,), type=type_)
    with pytest.raises(AnalysisError, match="recursive call path through 'root/helper/helper'"):
        check_program(root, root.entry_function())


@dataclass(frozen=True)
class _WriteGuardTarget(Target):
    name: ClassVar[str] = "test.occurrence.write.guard"
    algorithm: object = None

    def get_analyzer(self, selector):
        return self.algorithm


def test_an_analyzer_cannot_write_records_on_a_shared_derived_callee():
    def write_on_definition(_function, context):
        attach_metadata(_function.body, MemoryMetadata())

    target = _WriteGuardTarget(Analyzer("bad", write_on_definition, produces=(ComputeCostMetadata,)))
    root = replace(CrossModule, target=target, topologies=None)
    with pytest.raises(AnalysisError, match="changed Metadata it does not declare"):
        analyze(*check_program(root, root.entry_function()), analysis="bad")


def test_repeated_callee_temporaries_require_one_shared_memory_placement():
    type_ = TensorType((32,), DType.f32, None, StorageKind.GMEM)
    scalar = TensorType.scalar(DType.i32)
    zero = Constant(0, type=scalar)
    starts = Tuple((zero,), type=TupleType((scalar,)))
    shared_copy = ScheduleOp(op=Copy())
    shared_type = replace(type_, layout=Layout((32,), (1,)), storage=StorageKind.SMEM)

    def copy(value):
        destination = Call(Empty(type=shared_type), (), type=shared_type)
        return Call(shared_copy, (value, destination), type=shared_type)

    def write_back(value):
        out = Call(Zeros(type=type_), (), type=type_)
        return Call(InsertSlice(), (out, value, starts), type=type_)

    x, w = Var("x", type=type_), Var("w", type=type_, is_const=True)
    product = Call(Binary(kind=BinaryKind.MUL), (copy(x), copy(w)), type=type_)
    helper = Function.build(name="helper", params=(x, w), body=write_back(product), return_type=type_)
    root_x, root_w = Var("root_x", type=type_), Var("root_w", type=type_, is_const=True)
    cap_x, cap_w = Var("cap_x", type=type_), Var("cap_w", type=type_)
    first = Call(helper, (cap_x, cap_w), type=type_)
    second = Call(helper, (cap_x, cap_w), type=type_)
    result = write_back(Call(Binary(kind=BinaryKind.ADD), (copy(first), copy(second)), type=type_))
    topologies = (Topology("cta", 1), Topology("thread", 32))
    body = MeshRegion(
        Mesh(topologies, Layout((1, 32), (32, 1)), ("block", "lane")), result,
        params=(cap_x, cap_w), args=(root_x, root_w), type=type_,
    )
    function = Function.build(name="root", params=(root_x, root_w), body=body, return_type=type_)
    root = Module("SharedPressure", (helper, function), "root",
                  target=CudaTarget(_H200), topologies=topologies)
    analyzed = analyze(*check_program(root, function), analysis="memory")
    derived = next(fn for fn in analyzed.functions if fn.name == "helper")
    temporaries = [node for node in collect_exprs(derived.body)
                   if isinstance(node, Call) and isinstance(node.target, Empty)
                   and node.type.storage is StorageKind.SMEM]
    assert len(temporaries) == 2
    frame = CallFrame(analyzed.module, analyzed.function)
    calls = [child for node in collect_exprs(analyzed.function.body)
             if isinstance(node, Call) and (child := CallFrameVisitor.callee_frame(node, frame)) is not None
             and child.function is derived]
    assert len(calls) == 2
    assert calls[0].function is calls[1].function
    offsets = [get_metadata(node, MemoryMetadata).offsets for node in temporaries]
    assert all(len(held) == 1 for held in offsets)



def test_partial_indexed_loop_counts_only_reached_persistent_weights():
    root = _stacked_program(3, trips=2)
    result = analyze(*check_program(root, root.entry_function()), analysis="memory")
    peak = get_metadata(result.function, RegionMemoryMetadata).peak_for("gmem")
    assert (peak.persistent_bytes, peak.peak_bytes) == (48, 64)


def test_repeated_callee_placement_compares_alias_even_at_the_same_offset():
    type_ = TensorType((32,), DType.f32, None, StorageKind.GMEM)
    short = replace(type_, shape=(16,))
    scalar = TensorType.scalar(DType.i32)
    x, y = Var("x", type=type_), Var("y", type=short)
    offset = Var("offset", type=scalar)
    shared_type = replace(type_, layout=Layout((32,), (1,)), storage=StorageKind.SMEM)
    shared_short = replace(short, layout=Layout((16,), (1,)), storage=StorageKind.SMEM)
    held = Call(ScheduleOp(op=Copy()), (
        x, Call(Empty(type=shared_type), (), type=shared_type),
    ), type=shared_type)
    update = Call(ScheduleOp(op=Copy()), (
        y, Call(Empty(type=shared_short), (), type=shared_short),
    ), type=shared_short)
    starts = Tuple((offset,), type=TupleType((scalar,)))
    updated = Call(InsertSlice(), (held, update, starts), type=type_)
    value = Call(Binary(kind=BinaryKind.ADD), (updated, updated), type=type_)
    zero = Constant(0, type=scalar)
    result = Call(InsertSlice(), (
        Call(Zeros(type=type_), (), type=type_), value,
        Tuple((zero,), type=TupleType((scalar,))),
    ), type=type_)
    helper = Function.build(name="helper", params=(x, y, offset), body=result, return_type=type_)
    root_x, root_y = Var("root_x", type=type_), Var("root_y", type=short)
    cap_x, cap_y = Var("cap_x", type=type_), Var("cap_y", type=short)
    calls = tuple(Call(helper, (cap_x, cap_y, Constant(i, type=scalar)), type=type_) for i in (1, 2))
    values = Tuple(calls, type=TupleType((type_, type_)))
    topologies = (Topology("cta", 1), Topology("thread", 32))
    body = MeshRegion(
        Mesh(topologies, Layout((1, 32), (32, 1)), ("block", "lane")), values,
        params=(cap_x, cap_y), args=(root_x, root_y), type=values.type,
    )
    function = Function.build(name="root", params=(root_x, root_y), body=body, return_type=body.type)
    root = Module("SharedAliases", (helper, function), "root",
                  target=CudaTarget(_H200), topologies=topologies)
    analyzed = analyze(*check_program(root, function), analysis="memory")
    derived = next(fn for fn in analyzed.functions if fn.name == "helper")
    frame = CallFrame(analyzed.module, analyzed.function)
    reached = [
        child.function for node in collect_exprs(analyzed.function.body)
        if isinstance(node, Call)
        and (child := CallFrameVisitor.callee_frame(node, frame)) is not None
    ]
    assert reached == [derived, derived]
    buffers = [
        get_metadata(node, MemoryMetadata) for node in collect_exprs(derived.body)
        if isinstance(node, Call) and isinstance(node.target, Empty)
    ]
    assert len(buffers) == 2
    assert all(len(record.offsets) == 1 for record in buffers)


@pytest.mark.parametrize(
    "ownership,coordinates",
    [
        (("layers",), "{ [layer, addr] : 0 <= layer < 3 and addr = 2*layer }"),
        (
            ("layers", "experts"),
            "{ [layer, expert, addr] : 0 <= layer < 2 and 0 <= expert < 3 and addr = 10*layer + expert }",
        ),
    ],
)
def test_nested_weight_footprint_projection_keeps_instance_address_correlation(
    ownership, coordinates
):

    weight = Var("w", type=TensorType((32,), DType.f32, None, StorageKind.GMEM), is_const=True)
    item = ReachedAddresses(
        SSAValue(weight),
        None,
        DType.f32,
        isl.set(coordinates),
        AnalysisPrecision.EXACT,
        ownership=ownership,
    )
    footprint = Footprint((((), (item,)),))

    def project(source):
        shifted = source.relation.relation.apply_range(isl.map("{ [i] -> [j] : j=i+4 }"))
        return replace(
            source, relation=AccessRelation(shifted, source.relation.values)
        ), AnalysisPrecision.EXACT

    values = {
        BindingValuation: SimpleNamespace(project=project),
        LoopNestValuation: SimpleNamespace(domain=lambda: isl.set("{ [] }")),
    }
    result = resolve_footprint(
        footprint, SimpleNamespace(request_valuation=lambda kind: values[kind])
    )
    actual = result.groups[0][1][0]
    rank = len(ownership)
    # Shift only the address coordinate; instance coordinates remain correlated.
    offset = isl.map(
        "{ ["
        + ",".join(f"i{k}" for k in range(rank + 1))
        + "] -> ["
        + ",".join([*(f"i{k}" for k in range(rank)), f"i{rank}+4"])
        + "] }"
    )
    expected = item.reached.apply(offset)
    assert actual.reached.is_equal(expected)
    assert actual.must.is_equal(expected)
    assert actual.ownership == ownership
    assert not actual.values
