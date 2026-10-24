"""Scheduling lowering checks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import isl
import pytest

import tilefoundry.cli.schedule as schedule_cli
import tilefoundry.ir.hir.schedule as schedules
import tilefoundry.passes.transforms.convert_hir_to_tir as lowering_module
import tilefoundry.visitor_registry.instruction as instructions
from tests.core._corpus import load_program
from tests.core._source import import_dsl
from tests.nvidia.cuda.schedule._support import (
    _finalized,
    _RetainedStatements,
    _UnstatedInstruction,
    entry_function,
)
from tilefoundry.analysis.access import AliasKind
from tilefoundry.analysis.allocation import AllocationSummaryMetadata
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.metadata import (
    MemoryMetadata,
    RegionMemoryMetadata,
)
from tilefoundry.cli import main as cli_main
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import (
    Call,
    Var,
    detach_metadata,
    get_metadata,
)
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.empty import Empty
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.tir.memory import Copy, TensorView
from tilefoundry.ir.types import (
    ComposedLayout,
    DType,
    Layout,
    ShardLayout,
    StorageKind,
    Swizzle,
    TensorType,
    Topology,
)
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.passes.transforms.convert_hir_to_tir import LoweringError, TirBuilder
from tilefoundry.schedule import finalize
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    access_relation_registry,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.instruction import ScheduleRefusal, project
from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.ir.tir.nn.mma import TiledMma


def _m1_analysis():
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "wgmma_a_k_major.py"
    module = load_program(source)
    return (
        source,
        module,
        analyze(
            *check_program(module, module.entry_function()),
            analysis=("memory",),
        ),
    )


def _assert_cli_lowering_error(
    source: Path,
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "failed.py"
    assert cli_main(["schedule", "finalize", str(source), str(out)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    assert not out.exists()


def _returned_view(storage, chain=False):
    type_ = TensorType((32,), DType.f32, Layout((32,), (1,)), storage)
    param = Var("x", type=type_)
    function = Function.build(name="run", params=(param,), body=param, return_type=type_)
    leaf = Module("View", (function,), "run")
    slot = ModuleList(leaf)
    activation = replace(param)
    call = Call(slot, (activation,), type=type_)
    if chain:
        call = Call(slot, (call,), type=type_)
    root = Function.build(name="root", params=(activation,), body=call, return_type=type_)
    return Module(
        "ViewCaller",
        (root,),
        "root",
        children={"child": slot},
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 32)),
    )


def test_schedule_finalize_json_carries_the_same_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "wgmma_a_k_major.py"
    out = tmp_path / "finalized.json"

    assert cli_main(["schedule", "finalize", str(source), str(out), "--json"]) == 0
    assert capsys.readouterr() == ("", "")
    payload = json.loads(out.read_text())
    assert set(payload) == {"source"}
    python = tmp_path / "finalized.py"
    python.write_text(payload["source"])
    verify_prim_function(entry_function(load_program(python)))


def test_lowering_reports_smem_placement_above_capacity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = (
        Path(__file__).parents[1]
        / "fixtures"
        / "schedule"
        / "hir"
        / "gemm_8192x17408x5120_tma_store.py"
    )
    four_stage = tmp_path / source.name
    four_stage.write_text(source.read_text().replace("STAGES = 3", "STAGES = 4", 1))

    out = tmp_path / "finalized.py"
    assert cli_main(["schedule", "finalize", str(four_stage), str(out)]) == 0
    assert capsys.readouterr() == ("", "")
    module = load_program(four_stage)
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    memory = get_metadata(result.function, RegionMemoryMetadata)
    assert "smem placement peak 256.00KiB exceeds capacity 227.00KiB" in memory.errors
    verify_prim_function(entry_function(load_program(out)))


def test_lowering_rejects_instruction_without_access_relation(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    transfer = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and isinstance(expr.target, ScheduleOp)
        and not isinstance(expr.target.op, TiledMma)
    )
    monkeypatch.delitem(access_relation_registry._map, type(transfer.target.op))
    monkeypatch.setattr(schedule_cli, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(source, "states no access relations", tmp_path, capsys)


def test_lowering_rejects_nonidentity_inverse_for_a_written_return():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class WrittenReturnWindow:
    @func
    def gemm(x: Tensor[(1, 4), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as cta:
            out = tf.zeros(Tensor[(2, 4), "f32"])
            updated = tf.insert_slice(out, x, (1, 0))
            return updated[1:2, :]
""")
    with pytest.raises(
        lowering_module.LoweringError, match="requires a non-identity inverse relation"
    ):
        _finalized(module)


def test_zero_trip_carry_keeps_direct_address_proofs_and_the_loop():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ZeroTripCarry:
    @func
    def gemm(x: Tensor[(4,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                acc = tf.schedule((x, tf.empty(Tensor[(4,), "f32", "rmem"])), op=T.copy())
                for k in range(0):
                    acc = tf.schedule((x, tf.empty(Tensor[(4,), "f32", "rmem"])), op=T.copy())
                return tf.schedule((acc, tf.empty(Tensor[(4,), "f32"])), op=T.copy())
""")
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    summary = get_metadata(result.function, AllocationSummaryMetadata)
    carry = [
        proof
        for value in summary.values
        for proof in value.aliases
        if proof.kind in (AliasKind.CARRY_INIT, AliasKind.CARRY_YIELD)
    ]
    assert {proof.kind for proof in carry} == {AliasKind.CARRY_INIT, AliasKind.CARRY_YIELD}
    assert all(proof.relation.relation.is_identity() for proof in carry)
    lowered = _finalized(module)
    verify_module(lowered)
    verify_module(import_dsl(as_script(lowered)))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    assert len(visitor.loops) == 1
    loop = visitor.loops[0]
    assert tuple(static_dim_value(value) for value in (loop.start, loop.stop, loop.step)) == (
        0,
        0,
        1,
    )
    body = _RetainedStatements()
    body.visit(loop.body)
    assert sum(isinstance(op, Copy) for op in body.operations) == 1


def test_dynamic_window_keeps_a_mixed_scalar_and_constant_key():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class Window:
    @func
    def gemm(x: Tensor[(64, 8), "f32"], starts: Tensor[(1,), "i64"]) -> Tensor[(8, 8), "f32", "rmem"]:
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                held = tf.schedule((starts, tf.empty(Tensor[(1,), "i64", (1,), "rmem"])), op=T.copy())
                start = tf.reshape(held, new_shape=())
                window = x[start + 1:start + 9, 0:8]
                return tf.schedule((window, tf.empty(Tensor[(8, 8), "f32", (8, 8), "rmem"])), op=T.copy())
""")
    lowered = _finalized(module)
    verify_module(lowered)
    printed = as_script(lowered)
    verify_module(import_dsl(printed))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    views = [
        value
        for value in visitor.bindings.values()
        if isinstance(value, Call)
        and isinstance(value.target, TensorView)
        and value.type.shape == (8, 8)
    ]
    assert any(
        view.type.layout.strides == (8, 1)
        and any(
            isinstance(expr, Call)
            and isinstance(expr.target, Slice)
            and expr.target.sizes == (1, 1)
            and isinstance(expr.args[1].elements[0], Call)
            and expr.args[1].elements[1].value == 0
            for expr in collect_exprs(view)
        )
        for view in views
    )


def test_register_row_pointer_keeps_the_split_axis_whole():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 16)))
class RegisterRows:
    @func
    def gemm(x: Tensor[(16,16), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(16,), names=("col",)) as th:
                held = tf.schedule((x, tf.empty(Tensor[(16,16), "f32", (16,16 @ th.col), "rmem"])), op=T.copy())
                out = tf.zeros(Tensor[(16,16), "f32"])
                for i in range(16):
                    out = tf.insert_slice(out, held[i:i+1, :], (i,0))
                return out
""")
    lowered = _finalized(module)
    verify_module(lowered)
    verify_module(import_dsl(as_script(lowered)))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    slices = [
        expr
        for value in visitor.bindings.values()
        for expr in collect_exprs(value)
        if isinstance(expr, Call)
        and isinstance(expr.target, Slice)
        and isinstance(expr.args[0].type.layout, ShardLayout)
    ]
    assert slices and all(expr.target.sizes[1] == 16 for expr in slices)


@pytest.mark.parametrize("start,offset", [(0, 0), (64, 4096)])
def test_composed_view_keeps_its_layout_when_logical_rank_changes(start, offset):
    module = import_dsl(
        """
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ComposedReshape:
    @func
    def gemm(x: Tensor[(64,128), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                layout = ComposedLayout(Swizzle(3,4,3), 0, Layout(((8,8),(2,4,16)), ((512,64),(4096,16,1))))
                shared = tf.schedule((x, tf.empty(Tensor[(64,128), "bf16", layout, "smem"])), op=T.copy())
                reshaped = tf.reshape(shared[:,START:START+64], (8,8,16,4))
                return tf.schedule((reshaped, tf.empty(Tensor[(8,8,16,4), "bf16", "rmem"])), op=T.copy())
""".replace("START", str(start))
    )
    lowered = _finalized(module)
    verify_module(lowered)
    verify_module(import_dsl(as_script(lowered)))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    views = [
        value
        for value in visitor.bindings.values()
        if isinstance(value, Call)
        and isinstance(value.target, TensorView)
        and value.type.shape == (8, 8, 16, 4)
    ]
    assert any(
        value.type.layout
        == ComposedLayout(Swizzle(3, 4, 3), offset, Layout((8, 8, 16, 4), (512, 64, 4, 1)))
        for value in views
    )


@pytest.mark.parametrize(
    "width,start,shape,modes,strides,offset",
    [
        (256, 0, (16, 8, 16, 16), (16, 8, (4, 4), 16), (512, 64, (8192, 16), 1), 0),
        (32, 0, (16, 8, 2, 16), (16, 8, 2, 16), (512, 64, 16, 1), 0),
        (128, 128, (16, 8, 8, 16), (16, 8, (2, 4), 16), (512, 64, (8192, 16), 1), 16384),
    ],
)
def test_composed_reshape_retains_non_affine_outer_modes(
    width, start, shape, modes, strides, offset
):
    module = import_dsl(
        """
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ComposedReshape:
    @func
    def gemm(x: Tensor[(128,256), "bf16"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                layout = ComposedLayout(Swizzle(3,4,3), 0, Layout((128,(4,64)), (64,(8192,1))))
                shared = tf.schedule((x, tf.empty(Tensor[(128,256), "bf16", layout, "smem"])), op=T.copy())
                reshaped = tf.reshape(shared[:,START:START+WIDTH], SHAPE)
                return tf.schedule((reshaped, tf.empty(Tensor[SHAPE, "bf16", "rmem"])), op=T.copy())
""".replace("WIDTH", str(width))
        .replace("START", str(start))
        .replace("SHAPE", repr(shape))
    )
    lowered = _finalized(module)
    verify_module(import_dsl(as_script(lowered)))
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    expected = ComposedLayout(Swizzle(3, 4, 3), offset, Layout(modes, strides))
    assert any(
        isinstance(value, Call)
        and isinstance(value.target, TensorView)
        and value.type.layout == expected
        for value in visitor.bindings.values()
    )


def test_address_origin_can_divide_a_runtime_loop_index():
    module = import_dsl("""
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget
@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 1)))
class ModularWindow:
    @func
    def gemm(x: Tensor[(64,), "f32"]):
        with Mesh(("cta",), layout=(1,), names=("b",)) as cta:
            with Mesh(("thread",), layout=(1,), names=("t",)) as thread:
                out = tf.zeros(Tensor[(8,), "f32", "rmem"])
                for u in range(cta.b, 8, 1):
                    start = (u % 4) * 8
                    out = tf.schedule((x[start:start+8], tf.empty(Tensor[(8,), "f32", "rmem"])), op=T.copy())
                return out
""")
    verify_module(import_dsl(as_script(_finalized(module))))


@pytest.mark.parametrize("coordinate_division", [False, True])
def test_view_strides_distinguish_coordinate_and_parameter_divisions(coordinate_division):
    source = Var("x", type=TensorType((64,), DType.f32, Layout((64,), (1,)), StorageKind.GMEM))
    desired = replace(source.type, shape=(8,), layout=Layout((8,), (1,)))
    relation = AccessRelation(isl.map("{ [i] -> [floor(i/2)] : 0 <= i < 8 }"))
    if not coordinate_division:
        tile = Var("tile", type=TensorType.umat_scalar(DType.i64))
        relation = AccessRelation(
            isl.map("[tile] -> { [i] -> [i + floor(tile/4)] : 0 <= i < 8 }"), {"tile": tile}
        )
    function = Function.build(name="view", params=(source,), body=source, return_type=source.type)
    storage = lowering_module.StoragePlan(function)
    values = {} if coordinate_division else {id(tile): tile}
    if coordinate_division:
        with pytest.raises(lowering_module.LoweringError, match="non-affine address"):
            storage.view(source, relation, desired, values=values)
    else:
        view = storage.view(source, relation, desired, values=values)
        assert view.type.layout == Layout((8,), (1,))


def test_lowering_rejects_addressable_result_without_offsets(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    addressable = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and isinstance(expr.target, Empty)
        and expr.type.storage is StorageKind.SMEM
    )
    detach_metadata(addressable, MemoryMetadata)
    monkeypatch.setattr(schedule_cli, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(source, "addressable smem result but no offsets", tmp_path, capsys)


def test_lowering_rejects_unknown_hir_call(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    view = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and type(expr.target).__name__ == "Slice"
    )
    view.target = _UnstatedInstruction()
    monkeypatch.setattr(schedule_cli, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(
        source, "_UnstatedInstruction states no access relations", tmp_path, capsys
    )


def test_lowering_rejects_unscheduled_gmem_cast(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    cast = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, HirCast)
    )
    cast.type = replace(cast.type, storage=StorageKind.GMEM)
    monkeypatch.setattr(schedule_cli, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(
        source,
        "addressable gmem result but no offsets",
        tmp_path,
        capsys,
    )


def test_shared_projection_rejects_partial_atom_tiles() -> None:
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/wgmma_repeat_along_k.py"
    module = load_program(source)
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    schedule = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call)
        and isinstance(expr.target, ScheduleOp)
        and isinstance(expr.target.op, TiledMma)
    )
    acc, lhs, rhs = (arg.type for arg in schedule.args)
    with pytest.raises(
        ScheduleRefusal, match="iteration extent 17 is not divisible by single-issue extent 16"
    ):
        project(
            schedule.target.op,
            (replace(acc, shape=(acc.shape[0], 17)), lhs, replace(rhs, shape=(rhs.shape[0], 17))),
            None,
        )


@pytest.mark.parametrize("failure", ("leftmost", "factor"))
def test_grouped_projection_names_incompatible_repeat_mode(failure):
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/wgmma_n16_within_warpgroup_share.py"
    module = load_program(source)
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))
    schedule = next(
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call)
        and isinstance(value.target, ScheduleOp)
        and isinstance(value.target.op, TiledMma)
    )
    acc, lhs, rhs = (arg.type for arg in schedule.args)
    if failure == "leftmost":
        layout = replace(
            acc.layout, layout=Layout((4, 2, 8, 2, 2, 4, 6), (16, 8, 1, 3072, 1024, 128, 64))
        )
        operands = (replace(acc, layout=layout), lhs, rhs)
        reason = "acc axis 1: repeat is not leftmost at mode 4"
    else:
        layout = replace(
            rhs.layout,
            outer=Layout(((3, 4, 16), (3, 2, 2, 8)), ((6144, 16, 1), (8192, 1024, 512, 64))),
        )
        operands = (acc, lhs, replace(rhs, layout=layout))
        reason = "rhs iteration axis 1: repeat mode .* cannot factor as"
    with pytest.raises(ScheduleRefusal, match=reason):
        project(schedule.target.op, operands, None)


@pytest.mark.parametrize(
    ("name", "issues"), (("wgmma_repeat_along_n_order", 4), ("wgmma_k_slices_of_wide_run", 2))
)
def test_lowering_writes_static_issues_without_readmitting(name, issues, monkeypatch):
    source = Path(__file__).parents[1] / f"fixtures/schedule/hir/{name}.py"
    module = load_program(source)
    result = analyze(*check_program(module, module.entry_function()), analysis=("memory",))

    def reject(*args, **kwargs):
        raise AssertionError("lowering repeated instruction admission")

    monkeypatch.setattr(schedules, "check_call", reject)
    monkeypatch.setattr(instructions, "check_call", reject)
    lowered = lowering_module.Lowering(
        TypeInferContext(module=result.module, function=result.function, owner=result.function),
        result.function,
    ).run()
    statements = _RetainedStatements()
    statements.visit(lowered.body)
    assert not any(loop.induction_var.name.startswith("o_") for loop in statements.loops)
    issued = [stmt for stmt in statements.statements if isinstance(stmt.callable, TiledMma)]
    assert len(issued) == issues
    for stmt in issued:
        for arg in stmt.args:
            for expr in collect_exprs(statements.bindings.get(id(arg), arg)):
                if isinstance(expr, Call) and isinstance(expr.target, Slice):
                    assert all(
                        static_dim_value(start) is not None for start in expr.args[1].elements
                    )


@pytest.mark.parametrize("storage", [StorageKind.GMEM, StorageKind.SMEM])
@pytest.mark.parametrize("chain", [False, True])
def test_a_returned_view_passes_its_analyzed_storage_to_the_callee(storage, chain):
    lowered = _finalized(_returned_view(storage, chain))
    verify_module(lowered)
    function = lowered.entry_function()
    visitor = _RetainedStatements()
    visitor.visit(function.body)
    assert len(visitor.calls) == (2 if chain else 1)
    for prior in visitor.calls[:-1]:
        assert visitor.storage(prior.args[0]) is function.params[0]
        assert visitor.storage(prior.args[-1]) is function.params[0]
    call = visitor.calls[-1]
    assert call.args[0] is not call.args[-1]
    assert visitor.storage(call.args[0]) is function.params[0]
    assert visitor.storage(call.args[-1]) is function.params[-1]
    assert not any(isinstance(op, Copy) for op in visitor.operations)
    callee = _RetainedStatements()
    callee.visit(lowered.children[call.callable.key].elements[0].entry_function().body)
    assert not callee.operations


@pytest.mark.parametrize("read_old", [False, True])
@pytest.mark.parametrize("reshape", [False, True])
def test_inplace_output_reuses_storage_or_refuses_a_nonidentity_inverse(
    tmp_path, read_old, reshape
):
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    if read_old or reshape:
        text = source.read_text()
        if read_old:
            text = text.replace("(x, tf.empty", "(out[0:1, :], tf.empty")
        if reshape:
            text = text.replace(
                "return update(out, x)", "return tf.reshape(update(out, x), (4, 2))"
            )
        rewritten = tmp_path / source.name
        rewritten.write_text(text)
        source = rewritten
    if reshape:
        with pytest.raises(
            lowering_module.LoweringError, match="requires a non-identity inverse relation"
        ):
            _finalized(load_program(source))
        return
    lowered = _finalized(load_program(source))
    verify_module(lowered)
    function = lowered.entry_function()
    visitor = _RetainedStatements()
    visitor.visit(function.body)
    assert len(visitor.calls) == 1
    call = visitor.calls[0]
    assert call.args[0] is not call.args[-1]
    assert visitor.storage(call.args[0]) is function.params[0]
    assert visitor.storage(call.args[-1]) is function.params[-1]
    callee = _RetainedStatements()
    callee.visit(call.callable.body)
    copies = [stmt for stmt in callee.statements if isinstance(stmt.callable, Copy)]
    assert len(copies) == 2
    assert callee.storage(copies[0].args[0]) is call.callable.params[0 if read_old else 1]
    assert callee.storage(copies[-1].args[-1]) is call.callable.params[-1]
    assert not any(stmt.args[0].type.shape == (2, 4) for stmt in copies)


def test_read_only_call_before_an_update_keeps_the_input_storage(tmp_path):
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    text = (
        source.read_text()
        .replace(
            "    @func\n    def update",
            '    @func\n    def identity(out: Tensor[(2, 4), "f32"]):\n'
            "        return out\n\n    @func\n    def update",
        )
        .replace(
            "row = tf.schedule((x,",
            "old = identity(out)\n                row = tf.schedule((old[0:1, :],",
        )
        .replace("return tf.insert_slice(out,", "return tf.insert_slice(old,")
    )
    path = tmp_path / source.name
    path.write_text(text)
    lowered = _finalized(load_program(path))
    verify_module(lowered)
    visitor = _RetainedStatements()
    visitor.visit(lowered.entry_function().body)
    update = visitor.calls[0].callable
    body = _RetainedStatements()
    body.visit(update.body)
    assert len(body.calls) == 1
    identity = body.calls[0]
    assert body.storage(identity.args[0]) is update.params[0]
    assert body.storage(identity.args[-1]) is update.params[0]
    copies = [stmt for stmt in body.statements if isinstance(stmt.callable, Copy)]
    assert body.storage(copies[0].args[0]) is update.params[0]
    assert body.storage(copies[-1].args[-1]) is update.params[-1]


def test_staged_smem_keeps_logical_capacity_and_aligns_each_slot(tmp_path):
    source = (Path(__file__).parents[1] / "fixtures/schedule/hir/smem_padded_stages.py").read_text()
    path = tmp_path / "broadcast_stages.py"
    path.write_text(source.replace("((2, 3), (8, 1))", "((2, 3), (0, 1))"))
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    staged = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and getattr(expr.target, "buffers", 1) == 2
    )
    memory = get_metadata(staged, MemoryMetadata)
    assert len(memory.offsets) == 2
    assert all(offset % 16 == 0 for offset in memory.offsets)
    assert memory.buffer_bytes >= 2 * 3 * DType.f32.bit_width // 8


@pytest.mark.parametrize("later_use", [None, "callee", "caller"])
@pytest.mark.parametrize("nested", [False, True])
def test_insert_slice_and_retained_call_keep_parameter_storage(tmp_path, later_use, nested):
    source = (
        Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    ).read_text()
    if later_use == "callee":
        source = source.replace(
            "return tf.insert_slice(out, row, (1, 0))",
            "updated = tf.insert_slice(out, row, (1, 0))\n                return updated + out",
        )
    elif later_use == "caller":
        source = source.replace("return update(out, x)", "return update(out, x), out")
    if nested:
        source = source.replace(
            "    @func\n    def gemm",
            '    @func\n    def relay(out: Tensor[(2, 4), "f32"], x: Tensor[(1, 4), "f32"]):\n'
            "        return update(out, x)\n\n    @func\n    def gemm",
        )
        source = source.replace("return update(out, x), out", "return relay(out, x), out")
        source = source.replace("return update(out, x)  #", "return relay(out, x)  #")
    path = tmp_path / "update.py"
    path.write_text(source)
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    call = next(
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call) and isinstance(value.target, Function)
    )
    callee = call.target
    if nested:
        callee = next(
            value.target
            for value in collect_exprs(callee.body)
            if isinstance(value, Call) and isinstance(value.target, Function)
        )
    insert = next(
        value
        for value in collect_exprs(callee.body)
        if isinstance(value, Call) and isinstance(value.target, InsertSlice)
    )
    inserted = get_metadata(insert, MemoryMetadata)
    called = get_metadata(call, MemoryMetadata)
    proofs = {
        value.value: value for value in get_metadata(callee, AllocationSummaryMetadata).values
    }
    callers = {
        value.value: value
        for value in get_metadata(result.function, AllocationSummaryMetadata).values
    }
    if later_use == "callee":
        assert not proofs[SSAValue(insert)].aliases
    else:
        link = next(
            alias
            for alias in proofs[SSAValue(insert)].aliases
            if alias.kind is AliasKind.INSERT_SLICE
        )
        assert link.target.expr is insert.args[0]
        assert link.relation.relation.is_identity()
        composed = AllocationSummaryMetadata(tuple(proofs.values())).address_alias(
            proofs[SSAValue(insert)]
        )
        assert composed.target.expr is callee.params[0]
        assert composed.relation.relation.is_identity()
        assert inserted.buffer_bytes is None and inserted.offsets == ()
    if later_use is not None:
        assert not callers[SSAValue(call)].aliases
        assert called.offsets
    else:
        composed = AllocationSummaryMetadata(tuple(callers.values())).address_alias(
            callers[SSAValue(call)]
        )
        assert composed.target.expr is result.function.params[0]
        assert composed.relation.relation.is_identity()
        assert called.buffer_bytes is None and called.offsets == ()
    if later_use == "caller":
        with patch.object(
            TirBuilder, "transfer", autospec=True, side_effect=TirBuilder.transfer
        ) as copy:
            with pytest.raises(LoweringError, match="cannot be placed"):
                finalize(result, authored=module.entry_function())
        copy.assert_not_called()


def test_retained_update_translates_an_actual_window(tmp_path):
    source = (
        Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    ).read_text()
    source = source.replace("def gemm(out: Tensor[(2, 4)", "def gemm(out: Tensor[(3, 4)")
    source = source.replace("return update(out, x)", "return update(out[1:3, :], x)")
    path = tmp_path / "window.py"
    path.write_text(source)
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    call = next(
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call) and isinstance(value.target, Function)
    )
    memory = get_metadata(call, MemoryMetadata)
    values = {
        value.value: value
        for value in get_metadata(result.function, AllocationSummaryMetadata).values
    }
    composed = AllocationSummaryMetadata(tuple(values.values())).address_alias(
        values[SSAValue(call)]
    )
    assert composed.target.expr is result.function.params[0]
    expected = isl.map("{ [i, j] -> [i + 1, j] : 0 <= i < 2 and 0 <= j < 4 }")
    assert composed.relation.relation.is_equal(expected)
    assert memory.buffer_bytes is None and memory.offsets == ()


@pytest.mark.parametrize("returned", ["out", "out[1:2, :]"])
def test_read_only_call_return_keeps_storage_despite_later_actual_use(tmp_path, returned):
    source = (
        Path(__file__).parents[1] / "fixtures/schedule/hir/inplace_output_update.py"
    ).read_text()
    source = source.replace("return tf.insert_slice(out, row, (1, 0))", f"return {returned}")
    source = source.replace("return update(out, x)", "return update(out, x), out")
    path = tmp_path / "readonly.py"
    path.write_text(source)
    module = load_authored_ir(str(path))
    result = analyze(*check_program(module, module.entry_function()), analysis="memory")
    call = next(
        value
        for value in collect_exprs(result.function.body)
        if isinstance(value, Call) and isinstance(value.target, Function)
    )
    memory = get_metadata(call, MemoryMetadata)
    values = {
        value.value: value
        for value in get_metadata(result.function, AllocationSummaryMetadata).values
    }
    composed = AllocationSummaryMetadata(tuple(values.values())).address_alias(
        values[SSAValue(call)]
    )
    assert composed.target.expr is result.function.params[0]
    expected = (
        isl.map("{ [i, j] -> [i, j] : 0 <= i < 2 and 0 <= j < 4 }")
        if returned == "out"
        else isl.map("{ [i, j] -> [i + 1, j] : i = 0 and 0 <= j < 4 }")
    )
    assert composed.relation.relation.is_equal(expected)
    assert memory.buffer_bytes is None and memory.offsets == ()
