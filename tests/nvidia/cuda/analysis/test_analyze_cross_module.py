"""Analyze a root that reaches a child Module as one kernel.

A call into a child is device work in the current occurrence, so its cost lands
in the caller's totals and repeating the site or looping over it multiplies that
work rather than adding an occurrence. What the fixed-dimension query has to
establish first is that the two ends share one execution context.

No GPU, no codegen, no runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import pytest

from tests.core._corpus import Case, case, check_case, special
from tests.nvidia.cuda.models.qwen3_1_7b.case import CASE as QWEN_CASE
from tilefoundry import analysis, cli, target
from tilefoundry.dsl import *
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr, hir, types, utils
from tilefoundry.ir.hir import math, nn, tensor
from tilefoundry.ir.tir import memory
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda.models.qwen3_1_7b.model import Qwen3_1_7B

_H200 = "nvidia.h200_sxm"
_CTA = (Topology("cta", 132),)
FIXTURES = Path(__file__).parents[1] / "fixtures"
HIR_COMPOSITION = FIXTURES / "logical" / "hir_composition.py"
SCALED_MODULES = FIXTURES / "shapes" / "scaled_modules.py"
ANALYSIS = FIXTURES / "analysis"


def _matmul_records(
    result,
) -> tuple[tuple[analysis.ComputeCostMetadata, analysis.MemoryMetadata, analysis.RegionMemoryMetadata], ...]:
    """Both halves of each occurrence's MatMul record."""
    records = []

    def visit(frame):
        for node in core.collect_exprs(frame.function.body):
            if not isinstance(node, expr.Call):
                continue
            child = frame.callee_context(node)
            if child is not None:
                visit(child)
            elif isinstance(node.target, nn.MatMul):
                record = expr.get_metadata(node, analysis.ComputeCostMetadata)
                moved = expr.get_metadata(node, analysis.MemoryMetadata)
                assert record is not None
                records.append((record, moved))

    visit(valuation.Context(result.module, result.function))
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


def test_repeated_call_counts_twice() -> None:
    CrossModule = cli.load_authored_ir(f"{HIR_COMPOSITION}:CrossModule")
    _Twice = cli.load_authored_ir(f"{ANALYSIS / 'expert_called_twice.py'}:_Twice")
    one_result = analysis.analyze(
        *analysis.check_program(CrossModule, CrossModule.entry_function()),
        analysis="compute-cost",
    )
    two_result = analysis.analyze(*analysis.check_program(_Twice, _Twice.entry_function()), analysis="compute-cost")
    one_records = _matmul_records(one_result)
    two_records = _matmul_records(two_result)
    one = _flops(one_records)
    two = _flops(two_records)

    assert len(one_records) == 1 and two_records == one_records * 2
    assert one and two == {name: 2 * value for name, value in one.items()}


def test_looped_call_counts_per_trip() -> None:
    CrossModule = cli.load_authored_ir(f"{HIR_COMPOSITION}:CrossModule")
    _Looped = cli.load_authored_ir(f"{ANALYSIS / 'expert_called_in_loop.py'}:_Looped")
    one_result = analysis.analyze(
        *analysis.check_program(CrossModule, CrossModule.entry_function()),
        analysis="compute-cost",
    )
    looped_result = analysis.analyze(
        *analysis.check_program(_Looped, _Looped.entry_function()),
        analysis="compute-cost",
    )
    one_occurrence = _matmul_records(one_result)
    looped_occurrence = _matmul_records(looped_result)
    one_root = expr.get_metadata(one_result.function, analysis.ComputeCostMetadata)
    looped_root = expr.get_metadata(looped_result.function, analysis.ComputeCostMetadata)

    assert len(one_occurrence) == 1 and looped_occurrence == one_occurrence
    assert one_root is not None and looped_root is not None
    assert _flops((looped_root,)) == {
        name: 3 * value for name, value in _flops((one_root,)).items()
    }


def test_fused_weight_traffic_matches_callees() -> None:
    _Direct = cli.load_authored_ir(f"{ANALYSIS / 'direct_weight_matmul.py'}:_Direct")
    direct_result = analysis.analyze(
        *analysis.check_program(_Direct, _Direct.entry_function()),
        analysis=("compute-cost", "memory"),
    )
    CrossModule = cli.load_authored_ir(f"{HIR_COMPOSITION}:CrossModule")
    fused_result = analysis.analyze(
        *analysis.check_program(CrossModule, CrossModule.entry_function()),
        analysis=("compute-cost", "memory"),
    )
    direct = _traffic(_matmul_records(direct_result))
    fused = _traffic(_matmul_records(fused_result))

    assert direct and fused == direct


def test_child_hierarchy_mismatch_rejected() -> None:
    _Mismatch = cli.load_authored_ir(f"{ANALYSIS / 'child_on_another_hierarchy.py'}:_Mismatch")
    with pytest.raises(analysis.Error, match="different topology hierarchy") as caught:
        analysis.analyze(*analysis.check_program(_Mismatch, _Mismatch.entry_function()), analysis="compute-cost")
    assert "launch" not in str(caught.value)


class CrossModuleCase(Case):
    """A root reaching child Modules, analysed as one kernel."""


@case
class ChildOnCallerHierarchy(CrossModuleCase):
    """A child declaring the caller's hierarchy is accepted."""

    path: Path = ANALYSIS / "child_on_caller_hierarchy.py"
    name: str | None = "_Agreeing"

    def guard(self, state, problems):
        result = analysis.analyze(
            *analysis.check_program(state, state.entry_function(), dims={"child_topology_extent": 132}),
            analysis=("compute-cost", "memory"),
        )
        assert set(result.metadata_types) >= {
            analysis.ComputeCostMetadata,
            analysis.MemoryMetadata,
            analysis.RegionMemoryMetadata,
        }


class ReferenceProgram(CrossModuleCase):
    """The shared programs measure as one root each; what a call means is not restated."""

    def guard(self, state, problems):
        result = analysis.analyze(
            *analysis.check_program(state, state.entry_function()),
            analysis=("compute-cost", "memory"),
        )

        assert set(result.metadata_types) >= {
            analysis.ComputeCostMetadata,
            analysis.MemoryMetadata,
            analysis.RegionMemoryMetadata,
        }


@case
class SameModuleReference(ReferenceProgram):
    path: Path = HIR_COMPOSITION
    name: str | None = "SameModule"


@case
class CrossModuleReference(ReferenceProgram):
    path: Path = HIR_COMPOSITION
    name: str | None = "CrossModule"


@case
class FusedReference(ReferenceProgram):
    path: Path = HIR_COMPOSITION
    name: str | None = "Fused"


def _placed_primitives(fn) -> list[tuple[str, object]]:
    """Each costed primitive result in *fn* that a sliced Mesh reached."""
    found: list[tuple[str, object]] = []
    for node in core.collect_exprs(fn.body):
        if not isinstance(node, expr.Call) or isinstance(node.target, core.Function):
            continue
        for leaf in utils.tensor_types(node.type):
            if isinstance(leaf.layout, types.ShardLayout) and isinstance(
                leaf.layout.mesh.layout, types.ComposedLayout
            ):
                found.append((type(node.target).__name__, leaf.layout.mesh.layout))
    return found


@case
class MoEMegaKernelBranches(CrossModuleCase):
    """A lexical Mesh scope places nothing; the reshard into it is what does.

    So the branch is read from what its results retained: the sub-box the slice
    selected, as its own extent and origin rather than an extent alone.
    """

    path: Path = FIXTURES / "placed" / "moe_mega_kernel.py"
    name: str | None = "MoEMegaKernel"

    def guard(self, state, problems):
        expected = {"routed_expert": ((120,), 0), "shared_expert": ((12,), 120)}
        reached = {
            fn.name: _placed_primitives(fn)
            for frame in valuation.list_call_paths(analysis.check_program(state, state.entry_function())[0]).values()
            for fn in (frame.function,)
        }

        for branch, (shape, offset) in expected.items():
            placed = reached[branch]
            assert placed, branch
            assert {op for op, _ in placed} - {"Reshard"}, branch
            for _op, layout in placed:
                assert types.flatten(layout.outer).shape == shape
                assert layout.offset == offset


def _stacked_program(count, trips=None):
    """The ScaledChild stack with *count* layers whose loop reaches *trips* of them."""
    trips = count if trips is None else trips
    reached = "" if trips == count else f"_reaching_{trips}"
    return cli.load_authored_ir(f"{ANALYSIS / f'scaled_child_stack_{count}{reached}.py'}:Stack")


def _call_program(definition, *, count=1, indexed=False, name="run"):
    slot = ModuleList(definition, count) if indexed else ModuleList(definition)
    param = replace(definition.entry_function().params[0], is_const=False)
    calls = []
    value = param
    for index in range(count):
        tgt = slot
        if indexed:
            tgt = expr.Call(
                tensor.TupleGetItem(),
                (slot, expr.Constant(index, type=types.TensorType.from_scalar(expr.DType.i32))),
                type=slot.type.fields[index],
            )
        value = expr.Call(tgt, (value,), type=param.type)
        calls.append(value)
    if name == "twice":
        calls[1].args = (param,)
        value = expr.Call(math.Binary(kind=BinaryKind.ADD), tuple(calls), type=param.type)
    if name == "looped":
        carry = replace(param, name="carry")
        iteration = expr.Call(slot, (carry,), type=param.type)
        body = expr.Tuple((iteration,), type=types.TupleType((param.type,)))
        loop = core.LoopRegion(
            induction_var=expr.Var("i", type=types.TensorType.from_scalar(expr.DType.i32)),
            params=(carry,),
            args=(param,),
            body=body,
            yield_values=(iteration,),
            start=0,
            extent=3,
            step=1,
            type=body.type,
        )
        value = loop
    function = core.Function.from_parts(name=name, params=(param,), body=value, return_type=param.type)
    return core.Module(
        name,
        (function,),
        name,
        children={"layers" if indexed else "mm": slot},
        target=cuda.Target(_H200),
        topologies=_CTA,
    )


@pytest.mark.parametrize(
    "kind,flops,traffic,peak,persistent,occurrences",
    [
        ("cross", 512, 512, 512, 384, 1),
        ("twice", 1056, 1408, 640, 384, 2),
        ("looped", 1536, 1536, 640, 384, 1),
        ("one", 4, 48, 48, 32, 1),
        ("unrolled", 12, 144, 80, 64, 3),
        ("stacked", 12, 144, 80, 64, 1),
    ],
)
def test_retained_calls_match_acceptance_values(
    kind, flops, traffic, peak, persistent, occurrences
):
    Expert = cli.load_authored_ir(f"{HIR_COMPOSITION}:Expert")
    CrossModule = cli.load_authored_ir(f"{HIR_COMPOSITION}:CrossModule")
    ScaledChild = cli.load_authored_ir(f"{SCALED_MODULES}:ScaledChild")
    FusedScaledParent = cli.load_authored_ir(f"{SCALED_MODULES}:FusedScaledParent")
    StackedScaledParent = cli.load_authored_ir(f"{SCALED_MODULES}:StackedScaledParent")
    roots = {
        "cross": CrossModule,
        "twice": _call_program(Expert, count=2, name="twice"),
        "looped": _call_program(Expert, name="looped"),
        "one": replace(FusedScaledParent, topologies=_CTA),
        "unrolled": _call_program(ScaledChild, count=3, indexed=True),
        "stacked": replace(StackedScaledParent, topologies=_CTA),
    }
    root = roots[kind]
    result = analysis.analyze(
        *analysis.check_program(root, root.entry_function()),
        analysis=("compute-cost", "memory"),
    )
    cost = expr.get_metadata(result.function, analysis.ComputeCostMetadata)
    moved = expr.get_metadata(result.function, analysis.RegionMemoryMetadata)
    region = moved.peak_for("gmem")
    assert cost.flops.of("f32").total == flops
    assert moved.traffic.storage.of("gmem").total.total_bytes == traffic
    assert (region.peak_bytes, region.persistent_bytes) == (peak, persistent)
    assert len(result.functions) == 2
    frames = tuple(valuation.definitions(valuation.Context(result.module, result.function)))
    sites = [
        (call, child)
        for frame in frames
        for call in core.collect_exprs(frame.function.body)
        if isinstance(call, expr.Call) and (child := frame.callee_context(call)) is not None
    ]
    assert len(sites) == occurrences
    if kind == "stacked":
        assert (
            sum(isinstance(node, core.LoopRegion) for node in core.collect_exprs(result.function.body)) == 1
        )
    for call, child in sites:
        assert expr.get_metadata(child.function.body, analysis.ComputeCostMetadata) is not None
        assert expr.get_metadata(call, analysis.ComputeCostMetadata) == expr.get_metadata(
            child.function, analysis.ComputeCostMetadata
        )


def test_derived_size_ignores_instance_count():
    sizes = []
    for count in (3, 28):
        root = _stacked_program(count)
        checked, function = analysis.check_program(root, root.entry_function())
        functions = tuple(frame.function for frame in valuation.list_call_paths(checked).values())
        sizes.append((sum(len(core.collect_exprs(body.body)) for body in functions), len(functions)))
        assert sum(isinstance(node, core.LoopRegion) for node in core.collect_exprs(function.body)) == 1
        with pytest.raises(analysis.Error, match="derived body nodes.*node budget"):
            analysis.check_program(root, root.entry_function(), budget=sizes[-1][0] - 1)
    assert sizes[0] == sizes[1]
    assert sizes[0][1] == 2


def test_qwen_stack_shares_layer_definition():
    elements = Qwen3_1_7B.children["layers"].elements
    assert len(elements) == 28
    assert all(element is elements[0] for element in elements)
    root = QWEN_CASE.build()
    checked, _function = analysis.check_program(root, root.lookup("decoder_layer"), dims={"ctx_len": 128})
    assert (
        sum(len(core.collect_exprs(frame.function.body)) for frame in valuation.list_call_paths(checked).values())
        <= 10_000
    )


def _direct_program(callee, params, calls):
    body = expr.Tuple(tuple(calls), type=types.TupleType(tuple(call.type for call in calls)))
    function = core.Function.from_parts(name="root", params=params, body=body, return_type=body.type)
    return core.Module(
        "DirectCalls", (callee, function), "root", target=cuda.Target(_H200), topologies=_CTA
    )


@pytest.mark.parametrize("difference", ["layout", "storage", "mesh"])
def test_callee_rejects_different_signatures(difference):
    type_ = types.TensorType((4, 8), expr.DType.f32, None, StorageKind.UMAT)
    left = expr.Var("left", type=type_)
    changed = {
        "layout": replace(type_, layout=types.Layout((4, 8), (1, 4))),
        "storage": replace(type_, storage=StorageKind.SMEM),
        "mesh": type_,
    }[difference]
    right = expr.Var("right", type=changed)
    held = expr.Var("x", type=type_)
    callee = core.Function.from_parts(name="helper", params=(held,), body=held, return_type=type_)
    first = expr.Call(callee, (left,), type=type_)
    second = expr.Call(callee, (right,), type=changed)
    if difference == "mesh":
        first = core.MeshRegion(types.Mesh(_CTA, types.Layout((1,), (1,)), ("first",)), first, type=type_)
        second = core.MeshRegion(types.Mesh(_CTA, types.Layout((1,), (1,)), ("second",)), second, type=changed)
    root = _direct_program(callee, (left, right), (first, second))
    with pytest.raises(
        analysis.Error, match="callee 'helper'.*root/<value 0>.*root/<value 1>.*versus"
    ):
        analysis.check_program(root, root.entry_function())


def test_bodiless_and_recursive_callees_rejected():
    type_ = types.TensorType((4,), expr.DType.f32, None, StorageKind.UMAT)
    param = expr.Var("x", type=type_)
    callee = core.Function.from_parts(name="helper", params=(param,), body=None, return_type=type_)
    root = _direct_program(callee, (param,), (expr.Call(callee, (param,), type=type_),))
    with pytest.raises(expr.VerifyError, match="cannot analyze Function 'helper' without a body"):
        analysis.check_program(root, root.entry_function())
    callee.body = expr.Call(callee, (param,), type=type_)
    with pytest.raises(expr.VerifyError, match="recursive call path through 'root/helper/helper'"):
        analysis.check_program(root, root.entry_function())


@dataclass(frozen=True)
class _WriteGuardTarget(target.Target):
    name: ClassVar[str] = "test.occurrence.write.guard"
    algorithm: object = None

    def get_analyzer(self, selector):
        return self.algorithm


def test_undeclared_metadata_write_rejected():
    def write_on_definition(_function, context):
        _function.body.attach_metadata(analysis.MemoryMetadata())

    tgt = _WriteGuardTarget(
        target.Analyzer("bad", write_on_definition, produces=(analysis.ComputeCostMetadata,))
    )
    CrossModule = cli.load_authored_ir(f"{HIR_COMPOSITION}:CrossModule")
    root = replace(CrossModule, target=tgt, topologies=None)
    with pytest.raises(analysis.Error, match="changed Metadata it does not declare"):
        analysis.analyze(*analysis.check_program(root, root.entry_function()), analysis="bad")


def test_repeated_callee_shares_placement():
    type_ = types.TensorType((32,), expr.DType.f32, types.Layout((32,)), StorageKind.GMEM)
    scalar = types.TensorType.from_scalar(expr.DType.i32)
    zero = expr.Constant(0, type=scalar)
    starts = expr.Tuple((zero,), type=types.TupleType((scalar,)))
    shared_copy = hir.schedule.ScheduleOp(op=memory.Copy())
    shared_type = replace(type_, layout=types.Layout((32,), (1,)), storage=StorageKind.SMEM)

    def copy(value):
        destination = expr.Call(tensor.Empty(type=shared_type), (), type=shared_type)
        return expr.Call(shared_copy, (value, destination), type=shared_type)

    def write_back(value):
        out = expr.Call(tensor.Zeros(type=type_), (), type=type_)
        return expr.Call(tensor.InsertSlice(), (out, value, starts), type=type_)

    x, w = expr.Var("x", type=type_), expr.Var("w", type=type_, is_const=True)
    product = expr.Call(math.Binary(kind=BinaryKind.MUL), (copy(x), copy(w)), type=type_)
    helper = core.Function.from_parts(
        name="helper", params=(x, w), body=write_back(product), return_type=type_
    )
    root_x, root_w = expr.Var("root_x", type=type_), expr.Var("root_w", type=type_, is_const=True)
    cap_x, cap_w = expr.Var("cap_x", type=type_), expr.Var("cap_w", type=type_)
    first = expr.Call(helper, (cap_x, cap_w), type=type_)
    second = expr.Call(helper, (cap_x, cap_w), type=type_)
    result = write_back(expr.Call(math.Binary(kind=BinaryKind.ADD), (copy(first), copy(second)), type=type_))
    topologies = (Topology("cta", 1), Topology("thread", 32))
    body = core.MeshRegion(
        types.Mesh(topologies, types.Layout((1, 32), (32, 1)), ("block", "lane")),
        result,
        params=(cap_x, cap_w),
        args=(root_x, root_w),
        type=type_,
    )
    function = core.Function.from_parts(name="root", params=(root_x, root_w), body=body, return_type=type_)
    root = core.Module(
        "SharedPressure",
        (helper, function),
        "root",
        target=cuda.Target(_H200),
        topologies=topologies,
    )
    analyzed = analysis.analyze(*analysis.check_program(root, function), analysis="memory")
    derived = next(fn for fn in analyzed.functions if fn.name == "helper")
    temporaries = [
        node
        for node in core.collect_exprs(derived.body)
        if isinstance(node, expr.Call)
        and isinstance(node.target, tensor.Empty)
        and node.type.storage is StorageKind.SMEM
    ]
    assert len(temporaries) == 2
    frame = valuation.Context(analyzed.module, analyzed.function)
    calls = [
        child
        for node in core.collect_exprs(analyzed.function.body)
        if isinstance(node, expr.Call)
        and (child := frame.callee_context(node)) is not None
        and child.function is derived
    ]
    assert len(calls) == 2
    assert calls[0].function is calls[1].function
    offsets = [expr.get_metadata(node, analysis.MemoryMetadata).offsets for node in temporaries]
    assert all(len(held) == 1 for held in offsets)


def test_partial_loop_counts_reached_weights():
    root = _stacked_program(3, trips=2)
    result = analysis.analyze(*analysis.check_program(root, root.entry_function()), analysis="memory")
    peak = expr.get_metadata(result.function, analysis.RegionMemoryMetadata).peak_for("gmem")
    assert (peak.persistent_bytes, peak.peak_bytes) == (48, 64)


def test_callee_alias_placement_agrees():
    type_ = types.TensorType((32,), expr.DType.f32, types.Layout((32,)), StorageKind.GMEM)
    short = replace(type_, shape=(16,), layout=types.Layout((16,)))
    scalar = types.TensorType.from_scalar(expr.DType.i32)
    x, y = expr.Var("x", type=type_), expr.Var("y", type=short)
    offset = expr.Var("offset", type=scalar)
    shared_type = replace(type_, layout=types.Layout((32,), (1,)), storage=StorageKind.SMEM)
    shared_short = replace(short, layout=types.Layout((16,), (1,)), storage=StorageKind.SMEM)
    held = expr.Call(
        hir.schedule.ScheduleOp(op=memory.Copy()),
        (
            x,
            expr.Call(tensor.Empty(type=shared_type), (), type=shared_type),
        ),
        type=shared_type,
    )
    update = expr.Call(
        hir.schedule.ScheduleOp(op=memory.Copy()),
        (
            y,
            expr.Call(tensor.Empty(type=shared_short), (), type=shared_short),
        ),
        type=shared_short,
    )
    starts = expr.Tuple((offset,), type=types.TupleType((scalar,)))
    updated = expr.Call(tensor.InsertSlice(), (held, update, starts), type=type_)
    value = expr.Call(math.Binary(kind=BinaryKind.ADD), (updated, updated), type=type_)
    zero = expr.Constant(0, type=scalar)
    result = expr.Call(
        tensor.InsertSlice(),
        (
            expr.Call(tensor.Zeros(type=type_), (), type=type_),
            value,
            expr.Tuple((zero,), type=types.TupleType((scalar,))),
        ),
        type=type_,
    )
    helper = core.Function.from_parts(name="helper", params=(x, y, offset), body=result, return_type=type_)
    root_x, root_y = expr.Var("root_x", type=type_), expr.Var("root_y", type=short)
    cap_x, cap_y = expr.Var("cap_x", type=type_), expr.Var("cap_y", type=short)
    calls = tuple(
        expr.Call(helper, (cap_x, cap_y, expr.Constant(i, type=scalar)), type=type_) for i in (1, 2)
    )
    values = expr.Tuple(calls, type=types.TupleType((type_, type_)))
    topologies = (Topology("cta", 1), Topology("thread", 32))
    body = core.MeshRegion(
        types.Mesh(topologies, types.Layout((1, 32), (32, 1)), ("block", "lane")),
        values,
        params=(cap_x, cap_y),
        args=(root_x, root_y),
        type=values.type,
    )
    function = core.Function.from_parts(
        name="root", params=(root_x, root_y), body=body, return_type=body.type
    )
    root = core.Module(
        "SharedAliases", (helper, function), "root", target=cuda.Target(_H200), topologies=topologies
    )
    analyzed = analysis.analyze(*analysis.check_program(root, function), analysis="memory")
    derived = next(fn for fn in analyzed.functions if fn.name == "helper")
    frame = valuation.Context(analyzed.module, analyzed.function)
    reached = [
        child.function
        for node in core.collect_exprs(analyzed.function.body)
        if isinstance(node, expr.Call) and (child := frame.callee_context(node)) is not None
    ]
    assert reached == [derived, derived]
    buffers = [
        expr.get_metadata(node, analysis.MemoryMetadata)
        for node in core.collect_exprs(derived.body)
        if isinstance(node, expr.Call) and isinstance(node.target, tensor.Empty)
    ]
    assert len(buffers) == 2
    assert all(len(record.offsets) == 1 for record in buffers)


CASES = special(CrossModuleCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_cross_module_case(case: CrossModuleCase) -> None:
    check_case(case)
