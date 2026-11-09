"""Scheduling schedule typeinfer checks."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.nvidia.cuda.schedule._support import (
    _copy_schedule_call,
    _UnstatedInstruction,
)
from tilefoundry import inspection
from tilefoundry.interfaces import access, schedule, typeinfer
from tilefoundry.ir import expr, hir, pattern, tir, types
from tilefoundry.ir.hir import tensor


def test_arange_candidates_refuse_symbolic_start() -> None:
    type_ = types.TensorType((8,), expr.DType.i32, types.Layout((8,), (1,)), expr.StorageKind.RMEM)
    call = expr.Call(target=tensor.Arange(type=type_, start=expr.DimVar("start", 0, 8)), args=(), type=type_)
    with pytest.raises(schedule.RefusalError, match="start must be a static integer"):
        schedule.candidate_operands(call, tir.arange.Arange)


def test_single_issue_keeps_instruction_relations() -> None:
    sched = _copy_schedule_call(repeat=(1,), order=(0,))
    source = sched.args[0]
    destination_type = types.TensorType((4,), expr.DType.bf16, types.Layout((4,), (1,)), expr.StorageKind.SMEM)
    inferred = typeinfer.TypeInferContext(current_mesh=types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,))))
    typeinfer.inference_type(sched, inferred)
    scheduled = access.relations_of(sched, inferred)
    single = schedule.operand_relations(sched.target.op, (source.type, destination_type))
    assert len(scheduled) == len(single)
    assert all(
        left.relation.is_equal(right.relation)
        for left, right in zip(scheduled, single, strict=True)
    )


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (_copy_schedule_call(repeat=(2,)), "transfer tiling is not yet supported"),
        (_copy_schedule_call(order=(1,)), "is not a permutation"),
        (
            _copy_schedule_call(storage=expr.StorageKind.SMEM),
            "src:",
        ),
    ),
    ids=("repeat", "order", "operand-pattern"),
)
def test_schedule_typeinfer_rejects_invalid_contract(call: expr.Call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        typeinfer.inference_type(
            call,
            typeinfer.TypeInferContext(current_mesh=types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,)))),
        )


def test_schedule_typeinfer_requires_access_relations() -> None:
    type_ = types.TensorType((4,), expr.DType.bf16, types.Layout((4,), (1,)), expr.StorageKind.RMEM)
    call = expr.Call(
        target=hir.schedule.ScheduleOp(op=_UnstatedInstruction()),
        args=(expr.Var(name="value", type=type_),),
        type=type_,
    )
    with pytest.raises(ValueError, match="states no access relations"):
        typeinfer.inference_type(
            call,
            typeinfer.TypeInferContext(current_mesh=types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,)))),
        )


def test_schedule_typeinfer_requires_instruction_verifier(monkeypatch) -> None:
    type_ = types.TensorType((4,), expr.DType.bf16, types.Layout((4,), (1,)), expr.StorageKind.RMEM)
    call = expr.Call(
        target=hir.schedule.ScheduleOp(op=_UnstatedInstruction()),
        args=(expr.Var(name="value", type=type_),),
        type=type_,
    )
    monkeypatch.setitem(
        access.relation_registry.entries,
        _UnstatedInstruction,
        access.identity_relations,
    )

    context = typeinfer.TypeInferContext(current_mesh=types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,))))
    assert typeinfer.inference_type(call, replace(context, memo={})) == type_
    param = expr.schema_of(_UnstatedInstruction).signature[0]
    monkeypatch.setattr(param, "pattern", pattern.Tensor(shape=(3,)))
    with pytest.raises(
        ValueError, match="iteration extent 4 is not divisible by single-issue extent 3"
    ):
        typeinfer.inference_type(call, replace(context, memo={}))
    monkeypatch.setattr(param, "pattern", None)
    with pytest.raises(
        ValueError, match="has no input pattern, between relation, or registered verifier"
    ):
        typeinfer.inference_type(call, replace(context, memo={}))


def test_schedule_typeinfer_requires_whole_tiles(monkeypatch) -> None:
    type_ = types.TensorType((4,), expr.DType.bf16, types.Layout((4,), (1,)), expr.StorageKind.RMEM)
    call = expr.Call(
        target=hir.schedule.ScheduleOp(op=_UnstatedInstruction()),
        args=(expr.Var(name="value", type=type_),),
        type=type_,
    )
    param = expr.schema_of(_UnstatedInstruction).signature[0]
    monkeypatch.setattr(param, "pattern", pattern.Tensor(shape=(3,)))
    monkeypatch.setitem(
        access.relation_registry.entries,
        _UnstatedInstruction,
        access.identity_relations,
    )

    with pytest.raises(
        ValueError,
        match="iteration extent 4 is not divisible by single-issue extent 3",
    ):
        typeinfer.inference_type(
            call,
            typeinfer.TypeInferContext(current_mesh=types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,)))),
        )


def test_schedule_typeinfer_requires_instruction_operands(monkeypatch) -> None:
    type_ = types.TensorType((4,), expr.DType.bf16, types.Layout((4,), (1,)), expr.StorageKind.RMEM)
    call = expr.Call(
        target=hir.schedule.ScheduleOp(op=_UnstatedInstruction()),
        args=(),
        type=type_,
    )
    param = expr.schema_of(_UnstatedInstruction).signature[0]
    monkeypatch.setattr(param, "effect", expr.MemoryEffect.WRITE)
    monkeypatch.setattr(param, "pattern", None)

    with pytest.raises(
        ValueError,
        match="expects operands.*value.*got 0",
    ):
        typeinfer.inference_type(
            call,
            typeinfer.TypeInferContext(current_mesh=types.Mesh((types.Topology("thread", 32),), types.Layout((32,), (1,)))),
        )


def test_parameter_pattern_states_structure() -> None:
    declarations = {schema.op_class for schema in expr.iter_schemas() if schema.op_class is not None}
    for schema in expr.iter_schemas():
        if schema.op_class is None:
            continue
        stated = vars(schema.op_class).get("capability")
        capabilities = (stated,) if isinstance(stated, expr.OpCapability) else stated or ()
        declarations.update(
            capability.declaration
            for capability in capabilities
            if capability.declaration is not None
        )
    printer = inspection.PatternPrinter()
    for declaration in declarations:
        parameters = tuple(getattr(declaration, "parameters", ())) or expr.collect_param_defs(
            declaration
        )
        for parameter in parameters:
            if parameter.pattern is not None:
                assert printer.written(parameter.pattern, parameter.name) != parameter.name, (
                    f"{declaration.__name__}.{parameter.name} puts a predicate in its "
                    "structural pattern slot"
                )


def test_instruction_requires_execution_mesh(monkeypatch) -> None:
    monkeypatch.delattr(_UnstatedInstruction, "execution_mesh")
    with pytest.raises(ValueError, match="T.unstated_instruction execution_mesh must be"):
        pattern.declared_execution_mesh(_UnstatedInstruction)
