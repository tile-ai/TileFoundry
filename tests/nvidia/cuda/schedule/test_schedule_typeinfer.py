"""Scheduling schedule typeinfer checks."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.nvidia.cuda.schedule._support import (
    _copy_schedule_call,
    _UnstatedInstruction,
)
from tilefoundry.inspection import PatternPrinter
from tilefoundry.ir.core import (
    Call,
    OpCapability,
    Var,
)
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.ir.core.param_def import MemoryEffect, collect_param_defs
from tilefoundry.ir.hir.schedule import ScheduleOp, operand_relations
from tilefoundry.ir.hir.tensor.arange import Arange as HirArange
from tilefoundry.ir.pattern import (
    TensorPattern,
    declared_execution_mesh,
)
from tilefoundry.ir.tir.arange import Arange as TirArange
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    StorageKind,
    TensorType,
    Topology,
)
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    identity_relations,
    relations_of,
)
from tilefoundry.visitor_registry.candidates import candidate_operands
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.instruction import ScheduleRefusal
from tilefoundry.visitor_registry.typeinfer import inference_type


def test_arange_candidates_refuse_symbolic_start() -> None:
    type_ = TensorType((8,), DType.i32, Layout((8,), (1,)), StorageKind.RMEM)
    call = Call(target=HirArange(type=type_, start=DimVar("start", 0, 8)), args=(), type=type_)
    with pytest.raises(ScheduleRefusal, match="start must be a static integer"):
        candidate_operands(call, TirArange)


def test_single_issue_keeps_instruction_relations() -> None:
    schedule = _copy_schedule_call(repeat=(1,), order=(0,))
    source = schedule.args[0]
    destination_type = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.SMEM)
    inferred = TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,))))
    inference_type(schedule, inferred)
    scheduled = relations_of(schedule, inferred)
    single = operand_relations(schedule.target.op, (source.type, destination_type))
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
            _copy_schedule_call(storage=StorageKind.SMEM),
            "src:",
        ),
    ),
    ids=("repeat", "order", "operand-pattern"),
)
def test_schedule_typeinfer_rejects_invalid_contract(call: Call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        inference_type(
            call,
            TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,)))),
        )


def test_schedule_typeinfer_requires_access_relations() -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    call = Call(
        target=ScheduleOp(op=_UnstatedInstruction()),
        args=(Var(name="value", type=type_),),
        type=type_,
    )
    with pytest.raises(ValueError, match="states no access relations"):
        inference_type(
            call,
            TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,)))),
        )


def test_schedule_typeinfer_requires_instruction_verifier(monkeypatch) -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    call = Call(
        target=ScheduleOp(op=_UnstatedInstruction()),
        args=(Var(name="value", type=type_),),
        type=type_,
    )
    monkeypatch.setitem(
        access_relation_registry._map,
        _UnstatedInstruction,
        identity_relations,
    )

    context = TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,))))
    assert inference_type(call, replace(context, memo={})) == type_
    param = _UnstatedInstruction._op_schema.signature[0]
    monkeypatch.setattr(param, "pattern", TensorPattern(shape=(3,)))
    with pytest.raises(
        ValueError, match="iteration extent 4 is not divisible by single-issue extent 3"
    ):
        inference_type(call, replace(context, memo={}))
    monkeypatch.setattr(param, "pattern", None)
    with pytest.raises(
        ValueError, match="has no input pattern, between relation, or registered verifier"
    ):
        inference_type(call, replace(context, memo={}))


def test_schedule_typeinfer_requires_whole_tiles(monkeypatch) -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    call = Call(
        target=ScheduleOp(op=_UnstatedInstruction()),
        args=(Var(name="value", type=type_),),
        type=type_,
    )
    param = _UnstatedInstruction._op_schema.signature[0]
    monkeypatch.setattr(param, "pattern", TensorPattern(shape=(3,)))
    monkeypatch.setitem(
        access_relation_registry._map,
        _UnstatedInstruction,
        identity_relations,
    )

    with pytest.raises(
        ValueError,
        match="iteration extent 4 is not divisible by single-issue extent 3",
    ):
        inference_type(
            call,
            TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,)))),
        )


def test_schedule_typeinfer_requires_instruction_operands(monkeypatch) -> None:
    type_ = TensorType((4,), DType.bf16, Layout((4,), (1,)), StorageKind.RMEM)
    call = Call(
        target=ScheduleOp(op=_UnstatedInstruction()),
        args=(),
        type=type_,
    )
    param = _UnstatedInstruction._op_schema.signature[0]
    monkeypatch.setattr(param, "effect", MemoryEffect.WRITE)
    monkeypatch.setattr(param, "pattern", None)

    with pytest.raises(
        ValueError,
        match="expects operands.*value.*got 0",
    ):
        inference_type(
            call,
            TypeInferContext(current_mesh=Mesh((Topology("thread", 32),), Layout((32,), (1,)))),
        )


def test_parameter_pattern_states_structure() -> None:
    declarations = {schema.op_class for schema in iter_schemas() if schema.op_class is not None}
    for schema in iter_schemas():
        if schema.op_class is None:
            continue
        stated = vars(schema.op_class).get("capability")
        capabilities = (stated,) if isinstance(stated, OpCapability) else stated or ()
        declarations.update(
            capability.declaration
            for capability in capabilities
            if capability.declaration is not None
        )
    printer = PatternPrinter()
    for declaration in declarations:
        parameters = tuple(getattr(declaration, "parameters", ())) or collect_param_defs(
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
    with pytest.raises(ValueError, match="_UnstatedInstruction execution_mesh must be"):
        declared_execution_mesh(_UnstatedInstruction)
