"""Scheduling lowering checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import isl
import pytest

from tests.core._corpus import Case, case, check_case, reload, special
from tests.nvidia.cuda.schedule._support import (
    _finalized,
    _RetainedStatements,
    _UnstatedInstruction,
    entry_function,
)
from tilefoundry import analysis, cli, inspection, verifier
from tilefoundry.interfaces import access, schedule, typeinfer
from tilefoundry.ir import core, expr, hir, types
from tilefoundry.ir.hir import tensor
from tilefoundry.ir.tir import memory
from tilefoundry.passes import transforms
from tilefoundry.schedule import finalize
from tilefoundry_nvidia import cuda
from tilefoundry_nvidia.cuda.ir.tir import nn
from tilefoundry_nvidia.cuda.passes import transforms as passes_transforms

FIXTURES = Path(__file__).parents[1] / "fixtures"
VARIANTS = FIXTURES / "schedule" / "variants"
INPLACE = FIXTURES / "schedule" / "hir" / "inplace_output_update.py"

def _m1_analysis():
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "wgmma_a_k_major.py"
    module = cli.load_authored_ir(str(source))
    return (
        source,
        module,
        analysis.analyze(
            *analysis.check_program(module, module.entry_function()),
            analysis=("memory", "sync"),
        ),
    )


def _assert_cli_lowering_error(
    source: Path,
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "failed.py"
    assert cli.main(["schedule", "finalize", str(source), str(out)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    assert not out.exists()


def _returned_view(storage, chain=False):
    type_ = types.TensorType((32,), expr.DType.f32, types.Layout((32,), (1,)), storage)
    param = expr.Var("x", type=type_)
    function = core.Function.from_parts(name="run", params=(param,), body=param, return_type=type_)
    leaf = core.Module("View", (function,), "run")
    slot = core.FunctionRef(("child",), None)
    activation = replace(param)
    call = expr.Call(slot, (activation,), type=type_)
    if chain:
        call = expr.Call(slot, (call,), type=type_)
    root = core.Function.from_parts(name="root", params=(activation,), body=call, return_type=type_)
    return core.Module(
        "ViewCaller",
        (root,),
        "root",
        children={"child": leaf},
        target=cuda.Target("nvidia.h200_sxm"),
        topologies=(types.Topology("cta", 1), types.Topology("thread", 32)),
    )


def test_finalize_json_carries_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "wgmma_a_k_major.py"
    out = tmp_path / "finalized.json"

    assert cli.main(["schedule", "finalize", str(source), str(out), "--json"]) == 0
    assert capsys.readouterr() == ("", "")
    payload = json.loads(out.read_text())
    assert set(payload) == {"source"}
    verifier.verify_prim_function(entry_function(reload(payload["source"], "WGMMA_A_K_MAJOR")))


def test_lowering_reports_smem_overflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    four_stage = VARIANTS / "four_stage_gemm.py"

    out = tmp_path / "finalized.py"
    assert cli.main(["schedule", "finalize", str(four_stage), str(out)]) == 0
    assert capsys.readouterr() == ("", "")
    module = cli.load_authored_ir(str(four_stage))
    result = analysis.analyze(*analysis.check_program(module, module.entry_function()), analysis=("memory",))
    mem = expr.get_metadata(result.function, analysis.RegionMemoryMetadata)
    assert "smem placement peak 256.00KiB exceeds capacity 227.00KiB" in mem.errors
    verifier.verify_prim_function(entry_function(cli.load_authored_ir(str(out))))


def test_lowering_rejects_missing_access_relation(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    transfer = next(
        node
        for node in core.collect_exprs(result.function.body)
        if isinstance(node, expr.Call)
        and isinstance(node.target, hir.schedule.ScheduleOp)
        and not isinstance(node.target.op, nn.TiledMma)
    )
    monkeypatch.delitem(access.relation_registry.entries, type(transfer.target.op))
    monkeypatch.setattr(analysis, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(source, "states no access relations", tmp_path, capsys)


def test_finalize_rejects_missing_memory_analysis():
    module = cli.load_authored_ir(str(FIXTURES / "schedule" / "hir" / "smem_padded_stages.py"))
    result = analysis.analyze(
        *analysis.check_program(module, module.entry_function()), analysis=("compute-cost",)
    )
    with pytest.raises(transforms.LoweringError, match="finalize needs the result of analyze"):
        finalize(result)


def test_lowering_rejects_written_return_window():
    module = cli.load_authored_ir(str(FIXTURES / "diagnostics" / "lowering_written_return_window.py"))
    with pytest.raises(
        transforms.LoweringError, match="requires a non-identity inverse relation"
    ):
        _finalized(module)


class LoweringCase(Case):
    """A scheduling variant whose memory placement or lowered TIR keeps what it must."""


@case
class ZeroTripCarry(LoweringCase):
    path: Path = VARIANTS / "zero_trip_carry.py"

    def guard(self, state, problems):
        result = analysis.analyze(*analysis.check_program(state, state.entry_function()), analysis=("memory",))
        summary = expr.get_metadata(result.function, analysis.AllocationSummaryMetadata)
        carry = [
            proof
            for value in summary.values
            for proof in value.aliases
            if proof.kind in (analysis.AliasKind.CARRY_INIT, analysis.AliasKind.CARRY_YIELD)
        ]
        assert {proof.kind for proof in carry} == {analysis.AliasKind.CARRY_INIT, analysis.AliasKind.CARRY_YIELD}
        assert all(proof.relation.relation.is_identity() for proof in carry)
        lowered = _finalized(state)
        verifier.verify_module(lowered)
        verifier.verify_module(reload(inspection.as_script(lowered), lowered.name))
        visitor = _RetainedStatements()
        visitor.visit(lowered.lookup("gemm").body)
        assert len(visitor.loops) == 1
        loop = visitor.loops[0]
        assert tuple(expr.static_dim_value(value) for value in (loop.start, loop.stop, loop.step)) == (
            0,
            0,
            1,
        )
        body = _RetainedStatements()
        body.visit(loop.body)
        assert sum(isinstance(op, memory.Copy) for op in body.operations) == 1


@case
class DynamicWindow(LoweringCase):
    path: Path = VARIANTS / "dynamic_window.py"

    def guard(self, state, problems):
        lowered = _finalized(state)
        verifier.verify_module(lowered)
        printed = inspection.as_script(lowered)
        verifier.verify_module(reload(printed, lowered.name))
        visitor = _RetainedStatements()
        visitor.visit(lowered.lookup("gemm").body)
        views = [
            value
            for value in visitor.bindings.values()
            if isinstance(value, expr.Call)
            and isinstance(value.target, memory.TensorView)
            and value.type.shape == (8, 8)
        ]
        assert any(
            view.type.layout.strides == (8, 1)
            and any(
                isinstance(node, expr.Call)
                and isinstance(node.target, tensor.Slice)
                and node.target.sizes == (1, 1)
                and isinstance(node.args[1].elements[0], expr.Call)
                and node.args[1].elements[1].value == 0
                for node in core.collect_exprs(view)
            )
            for view in views
        )


@case
class RegisterRows(LoweringCase):
    path: Path = VARIANTS / "register_rows.py"

    def guard(self, state, problems):
        lowered = _finalized(state)
        verifier.verify_module(lowered)
        verifier.verify_module(reload(inspection.as_script(lowered), lowered.name))
        visitor = _RetainedStatements()
        visitor.visit(lowered.lookup("gemm").body)
        slices = [
            node
            for value in visitor.bindings.values()
            for node in core.collect_exprs(value)
            if isinstance(node, expr.Call)
            and isinstance(node.target, tensor.Slice)
            and isinstance(node.args[0].type.layout, types.ShardLayout)
        ]
        assert slices and all(node.target.sizes[1] == 16 for node in slices)


@dataclass(frozen=True)
class ComposedRankView(LoweringCase):
    offset: int = 0

    def guard(self, state, problems):
        lowered = _finalized(state)
        verifier.verify_module(lowered)
        verifier.verify_module(reload(inspection.as_script(lowered), lowered.name))
        visitor = _RetainedStatements()
        visitor.visit(lowered.lookup("gemm").body)
        views = [
            value
            for value in visitor.bindings.values()
            if isinstance(value, expr.Call)
            and isinstance(value.target, memory.TensorView)
            and value.type.shape == (8, 8, 16, 4)
        ]
        assert any(
            value.type.layout
            == types.ComposedLayout(types.Swizzle(3, 4, 3), self.offset, types.Layout((8, 8, 16, 4), (512, 64, 4, 1)))
            for value in views
        )


@case
class ComposedRankLeftHalf(ComposedRankView):
    path: Path = VARIANTS / "composed_rank_left_half.py"


@case
class ComposedRankRightHalf(ComposedRankView):
    path: Path = VARIANTS / "composed_rank_right_half.py"
    offset: int = 4096


@dataclass(frozen=True)
class ComposedOuterReshape(LoweringCase):
    modes: tuple = ()
    strides: tuple = ()
    offset: int = 0

    def guard(self, state, problems):
        lowered = _finalized(state)
        verifier.verify_module(reload(inspection.as_script(lowered), lowered.name))
        visitor = _RetainedStatements()
        visitor.visit(lowered.lookup("gemm").body)
        expected = types.ComposedLayout(types.Swizzle(3, 4, 3), self.offset, types.Layout(self.modes, self.strides))
        assert any(
            isinstance(value, expr.Call)
            and isinstance(value.target, memory.TensorView)
            and value.type.layout == expected
            for value in visitor.bindings.values()
        )


@case
class ComposedOuterFullWidth(ComposedOuterReshape):
    path: Path = VARIANTS / "composed_outer_full_width.py"
    modes: tuple = (16, 8, (4, 4), 16)
    strides: tuple = (512, 64, (8192, 16), 1)


@case
class ComposedOuterWithinGroup(ComposedOuterReshape):
    path: Path = VARIANTS / "composed_outer_within_group.py"
    modes: tuple = (16, 8, 2, 16)
    strides: tuple = (512, 64, 16, 1)


@case
class ComposedOuterRightHalf(ComposedOuterReshape):
    path: Path = VARIANTS / "composed_outer_right_half.py"
    modes: tuple = (16, 8, (2, 4), 16)
    strides: tuple = (512, 64, (8192, 16), 1)
    offset: int = 16384


@case
class ModularWindow(LoweringCase):
    path: Path = VARIANTS / "modular_window.py"

    def guard(self, state, problems):
        lowered = _finalized(state)
        verifier.verify_module(reload(inspection.as_script(lowered), lowered.name))


@pytest.mark.parametrize("coordinate_division", [False, True])
def test_view_strides_distinguish_divisions(coordinate_division):
    source = expr.Var("x", type=types.TensorType((64,), expr.DType.f32, types.Layout((64,), (1,)), expr.StorageKind.GMEM))
    desired = replace(source.type, shape=(8,), layout=types.Layout((8,), (1,)))
    relation = access.Relation(isl.map("{ [i] -> [floor(i/2)] : 0 <= i < 8 }"))
    if not coordinate_division:
        tile = expr.Var("tile", type=types.TensorType.from_umat_scalar(expr.DType.i64))
        relation = access.Relation(
            isl.map("[tile] -> { [i] -> [i + floor(tile/4)] : 0 <= i < 8 }"), {"tile": tile}
        )
    function = core.Function.from_parts(name="view", params=(source,), body=source, return_type=source.type)
    storage = transforms.StoragePlan(function)
    values = {} if coordinate_division else {id(tile): tile}
    if coordinate_division:
        with pytest.raises(transforms.LoweringError, match="non-affine address"):
            storage.view(source, relation, desired, values=values)
    else:
        view = storage.view(source, relation, desired, values=values)
        assert view.type.layout == types.Layout((8,), (1,))


def test_lowering_rejects_offsetless_result(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    addressable = next(
        node
        for node in core.collect_exprs(result.function.body)
        if isinstance(node, expr.Call)
        and isinstance(node.target, tensor.Empty)
        and node.type.storage is expr.StorageKind.SMEM
    )
    addressable.detach_metadata(analysis.MemoryMetadata)
    monkeypatch.setattr(analysis, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(source, "addressable smem result but no offsets", tmp_path, capsys)


def test_lowering_rejects_unknown_hir_call(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    view = next(
        node
        for node in core.collect_exprs(result.function.body)
        if isinstance(node, expr.Call) and type(node.target).__name__ == "Slice"
    )
    view.target = _UnstatedInstruction()
    monkeypatch.setattr(analysis, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(
        source, "_UnstatedInstruction states no access relations", tmp_path, capsys
    )


def test_lowering_rejects_unscheduled_gmem_cast(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _module, result = _m1_analysis()
    cast = next(
        node
        for node in core.collect_exprs(result.function.body)
        if isinstance(node, expr.Call) and isinstance(node.target, tensor.Cast)
    )
    cast.type = replace(cast.type, storage=expr.StorageKind.GMEM)
    monkeypatch.setattr(analysis, "analyze", lambda *_args, **_kwargs: result)

    _assert_cli_lowering_error(
        source,
        "addressable gmem result but no offsets",
        tmp_path,
        capsys,
    )


def test_projection_rejects_partial_atom_tiles() -> None:
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/wgmma_repeat_along_k.py"
    module = cli.load_authored_ir(str(source))
    result = analysis.analyze(*analysis.check_program(module, module.entry_function()), analysis=("memory",))
    sched = next(
        node
        for node in core.collect_exprs(result.function.body)
        if isinstance(node, expr.Call)
        and isinstance(node.target, hir.schedule.ScheduleOp)
        and isinstance(node.target.op, nn.TiledMma)
    )
    acc, lhs, rhs = (arg.type for arg in sched.args)
    with pytest.raises(
        schedule.RefusalError, match="iteration extent 17 is not divisible by single-issue extent 16"
    ):
        schedule.project(
            sched.target.op,
            (replace(acc, shape=(acc.shape[0], 17)), lhs, replace(rhs, shape=(rhs.shape[0], 17))),
            None,
        )


@pytest.mark.parametrize("failure", ("leftmost", "factor"))
def test_projection_names_bad_repeat_mode(failure):
    source = Path(__file__).parents[1] / "fixtures/schedule/hir/wgmma_n16_within_warpgroup_share.py"
    module = cli.load_authored_ir(str(source))
    result = analysis.analyze(*analysis.check_program(module, module.entry_function()), analysis=("memory",))
    sched = next(
        value
        for value in core.collect_exprs(result.function.body)
        if isinstance(value, expr.Call)
        and isinstance(value.target, hir.schedule.ScheduleOp)
        and isinstance(value.target.op, nn.TiledMma)
    )
    acc, lhs, rhs = (arg.type for arg in sched.args)
    if failure == "leftmost":
        layout = replace(
            acc.layout, layout=types.Layout((4, 2, 8, 2, 2, 4, 6), (16, 8, 1, 3072, 1024, 128, 64))
        )
        operands = (replace(acc, layout=layout), lhs, rhs)
        reason = "acc axis 1: repeat is not leftmost at mode 4"
    else:
        layout = replace(
            rhs.layout,
            outer=types.Layout(((3, 4, 16), (3, 2, 2, 8)), ((6144, 16, 1), (8192, 1024, 512, 64))),
        )
        operands = (acc, lhs, replace(rhs, layout=layout))
        reason = "rhs iteration axis 1: repeat mode .* cannot factor as"
    with pytest.raises(schedule.RefusalError, match=reason):
        schedule.project(sched.target.op, operands, None)


@pytest.mark.parametrize(
    ("name", "issues"), (("wgmma_repeat_along_n_order", 4), ("wgmma_k_slices_of_wide_run", 2))
)
def test_lowering_issues_statically_without_readmission(name, issues, monkeypatch):
    source = Path(__file__).parents[1] / f"fixtures/schedule/hir/{name}.py"
    module = cli.load_authored_ir(str(source))
    result = analysis.analyze(
        *analysis.check_program(module, module.entry_function()), analysis=("memory", "sync")
    )

    def reject(*args, **kwargs):
        raise AssertionError("lowering repeated instruction admission")

    monkeypatch.setattr(schedule, "check_call", reject)
    monkeypatch.setattr(schedule.instruction, "check_call", reject)
    lowered = passes_transforms.CudaLowering(
        typeinfer.TypeInferContext(module=result.module, function=result.function, owner=result.function),
    ).run()
    statements = _RetainedStatements()
    statements.visit(lowered.body)
    assert not any(loop.induction_var.name.startswith("o_") for loop in statements.loops)
    issued = [stmt for stmt in statements.statements if isinstance(stmt.callable, nn.TiledMma)]
    assert len(issued) == issues
    for stmt in issued:
        for arg in stmt.args:
            for node in core.collect_exprs(statements.bindings.get(id(arg), arg)):
                if isinstance(node, expr.Call) and isinstance(node.target, tensor.Slice):
                    assert all(
                        expr.static_dim_value(start) is not None for start in node.args[1].elements
                    )


@pytest.mark.parametrize("storage", [expr.StorageKind.GMEM, expr.StorageKind.SMEM])
@pytest.mark.parametrize("chain", [False, True])
def test_returned_view_passes_storage(storage, chain):
    lowered = _finalized(_returned_view(storage, chain))
    verifier.verify_module(lowered)
    function = lowered.lookup("root")
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
    assert not any(isinstance(op, memory.Copy) for op in visitor.operations)
    callee = _RetainedStatements()
    callee.visit(core.resolve(lowered, call.callable)[1].body)
    assert not callee.operations


@pytest.mark.parametrize("read_old", [False, True])
@pytest.mark.parametrize("reshape", [False, True])
def test_inplace_output_reuses_or_refuses(read_old, reshape):
    source = {
        (False, False): INPLACE,
        (True, False): VARIANTS / "inplace_reads_old_output.py",
        (False, True): FIXTURES / "diagnostics" / "lowering_reshaped_retained_update.py",
        (True, True): FIXTURES / "diagnostics" / "lowering_reshaped_update_reads_old_output.py",
    }[read_old, reshape]
    if reshape:
        with pytest.raises(
            transforms.LoweringError, match="requires a non-identity inverse relation"
        ):
            _finalized(cli.load_authored_ir(str(source)))
        return
    lowered = _finalized(cli.load_authored_ir(str(source)))
    verifier.verify_module(lowered)
    function = lowered.lookup("gemm")
    visitor = _RetainedStatements()
    visitor.visit(function.body)
    assert len(visitor.calls) == 1
    call = visitor.calls[0]
    assert call.args[0] is not call.args[-1]
    assert visitor.storage(call.args[0]) is function.params[0]
    assert visitor.storage(call.args[-1]) is function.params[-1]
    retained = core.resolve(lowered, call.callable)[1]
    callee = _RetainedStatements()
    callee.visit(retained.body)
    copies = [stmt for stmt in callee.statements if isinstance(stmt.callable, memory.Copy)]
    assert len(copies) == 2
    assert callee.storage(copies[0].args[0]) is retained.params[0 if read_old else 1]
    assert callee.storage(copies[-1].args[-1]) is retained.params[-1]
    assert not any(stmt.args[0].type.shape == (2, 4) for stmt in copies)


@case
class ReadBeforeUpdate(LoweringCase):
    path: Path = VARIANTS / "read_before_update.py"

    def guard(self, state, problems):
        lowered = _finalized(state)
        verifier.verify_module(lowered)
        visitor = _RetainedStatements()
        visitor.visit(lowered.lookup("gemm").body)
        update = core.resolve(lowered, visitor.calls[0].callable)[1]
        body = _RetainedStatements()
        body.visit(update.body)
        assert len(body.calls) == 1
        identity = body.calls[0]
        assert body.storage(identity.args[0]) is update.params[0]
        assert body.storage(identity.args[-1]) is update.params[0]
        copies = [stmt for stmt in body.statements if isinstance(stmt.callable, memory.Copy)]
        assert body.storage(copies[0].args[0]) is update.params[0]
        assert body.storage(copies[-1].args[-1]) is update.params[-1]


@case
class BroadcastStages(LoweringCase):
    path: Path = VARIANTS / "broadcast_stages.py"

    def guard(self, state, problems):
        result = analysis.analyze(*analysis.check_program(state, state.entry_function()), analysis="memory")
        staged = next(
            node
            for node in core.collect_exprs(result.function.body)
            if isinstance(node, expr.Call) and getattr(node.target, "buffers", 1) == 2
        )
        mem = expr.get_metadata(staged, analysis.MemoryMetadata)
        assert len(mem.offsets) == 2
        assert all(offset % 16 == 0 for offset in mem.offsets)
        assert mem.buffer_bytes >= 2 * 3 * expr.DType.f32.bit_width // 8


@pytest.mark.parametrize("later_use", [None, "callee", "caller"])
@pytest.mark.parametrize("nested", [False, True])
def test_retained_insert_keeps_parameter_storage(later_use, nested):
    path = {
        (None, False): INPLACE,
        (None, True): VARIANTS / "retained_update_via_relay.py",
        ("callee", False): VARIANTS / "retained_callee_reuses_output.py",
        ("callee", True): VARIANTS / "retained_callee_reuses_output_via_relay.py",
        ("caller", False): VARIANTS / "retained_caller_returns_output.py",
        ("caller", True): VARIANTS / "retained_caller_returns_output_via_relay.py",
    }[later_use, nested]
    module = cli.load_authored_ir(str(path))
    checked, root = analysis.check_program(module, module.entry_function())
    result = analysis.analyze(checked, root, analysis=("memory", "sync"))
    call = next(
        value
        for value in core.collect_exprs(result.function.body)
        if isinstance(value, expr.Call) and isinstance(value.target, core.FunctionRef)
    )
    callee = core.resolve(checked, call.target)[1]
    if nested:
        callee = core.resolve(checked, next(
            value.target
            for value in core.collect_exprs(callee.body)
            if isinstance(value, expr.Call) and isinstance(value.target, core.FunctionRef)
        ))[1]
    insert = next(
        value
        for value in core.collect_exprs(callee.body)
        if isinstance(value, expr.Call) and isinstance(value.target, tensor.InsertSlice)
    )
    inserted = expr.get_metadata(insert, analysis.MemoryMetadata)
    called = expr.get_metadata(call, analysis.MemoryMetadata)
    proofs = {
        value.value: value for value in expr.get_metadata(callee, analysis.AllocationSummaryMetadata).values
    }
    callers = {
        value.value: value
        for value in expr.get_metadata(result.function, analysis.AllocationSummaryMetadata).values
    }
    if later_use == "callee":
        assert not proofs[core.SSAValue(insert)].aliases
    else:
        link = next(
            alias
            for alias in proofs[core.SSAValue(insert)].aliases
            if alias.kind is analysis.AliasKind.INSERT_SLICE
        )
        assert link.target.expr is insert.args[0]
        assert link.relation.relation.is_identity()
        composed = analysis.AllocationSummaryMetadata(tuple(proofs.values())).address_alias(
            proofs[core.SSAValue(insert)]
        )
        assert composed.target.expr is callee.params[0]
        assert composed.relation.relation.is_identity()
        assert inserted.buffer_bytes is None and inserted.offsets == ()
    if later_use is not None:
        assert not callers[core.SSAValue(call)].aliases
        assert called.offsets
    else:
        composed = analysis.AllocationSummaryMetadata(tuple(callers.values())).address_alias(
            callers[core.SSAValue(call)]
        )
        assert composed.target.expr is result.function.params[0]
        assert composed.relation.relation.is_identity()
        assert called.buffer_bytes is None and called.offsets == ()
    if later_use == "caller":
        with patch.object(
            transforms.TirBuilder, "transfer", autospec=True, side_effect=transforms.TirBuilder.transfer
        ) as copy:
            with pytest.raises(transforms.LoweringError, match="cannot be placed"):
                finalize(result)
        copy.assert_not_called()


@case
class RetainedWindow(LoweringCase):
    path: Path = VARIANTS / "retained_window.py"

    def guard(self, state, problems):
        result = analysis.analyze(*analysis.check_program(state, state.entry_function()), analysis="memory")
        call = next(
            value
            for value in core.collect_exprs(result.function.body)
            if isinstance(value, expr.Call) and isinstance(value.target, core.FunctionRef)
        )
        mem = expr.get_metadata(call, analysis.MemoryMetadata)
        values = {
            value.value: value
            for value in expr.get_metadata(result.function, analysis.AllocationSummaryMetadata).values
        }
        composed = analysis.AllocationSummaryMetadata(tuple(values.values())).address_alias(
            values[core.SSAValue(call)]
        )
        assert composed.target.expr is result.function.params[0]
        expected = isl.map("{ [i, j] -> [i + 1, j] : 0 <= i < 2 and 0 <= j < 4 }")
        assert composed.relation.relation.is_equal(expected)
        assert mem.buffer_bytes is None and mem.offsets == ()


@dataclass(frozen=True)
class ReadOnlyReturn(LoweringCase):
    relation: str = ""

    def guard(self, state, problems):
        result = analysis.analyze(*analysis.check_program(state, state.entry_function()), analysis="memory")
        call = next(
            value
            for value in core.collect_exprs(result.function.body)
            if isinstance(value, expr.Call) and isinstance(value.target, core.FunctionRef)
        )
        mem = expr.get_metadata(call, analysis.MemoryMetadata)
        values = {
            value.value: value
            for value in expr.get_metadata(result.function, analysis.AllocationSummaryMetadata).values
        }
        composed = analysis.AllocationSummaryMetadata(tuple(values.values())).address_alias(
            values[core.SSAValue(call)]
        )
        assert composed.target.expr is result.function.params[0]
        assert composed.relation.relation.is_equal(isl.map(self.relation))
        assert mem.buffer_bytes is None and mem.offsets == ()


@case
class ReadOnlyWholeReturn(ReadOnlyReturn):
    path: Path = VARIANTS / "readonly_whole.py"
    relation: str = "{ [i, j] -> [i, j] : 0 <= i < 2 and 0 <= j < 4 }"


@case
class ReadOnlyWindowReturn(ReadOnlyReturn):
    path: Path = VARIANTS / "readonly_window.py"
    relation: str = "{ [i, j] -> [i + 1, j] : i = 0 and 0 <= j < 4 }"


CASES = special(LoweringCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_lowering_case(case: LoweringCase) -> None:
    check_case(case)
