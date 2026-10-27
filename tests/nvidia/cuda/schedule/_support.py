"""Shared scheduling assertions and unit-test program builders."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from math import prod
from pathlib import Path

import torch

from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import (
    Call,
    Constant,
    Op,
    Tuple,
    Var,
)
from tilefoundry.ir.core.module import Module, ModuleList
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.tensor.empty import Empty
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.pattern import (
    is_ranked_tensor,
    thread_execution_mesh,
)
from tilefoundry.ir.tir import PrimFunction
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.memory import PtrOf, TensorView
from tilefoundry.ir.tir.stmts import Evaluate, For, If, LetStmt, MeshScope, Sequential, While
from tilefoundry.ir.types import (
    DType,
    Layout,
    StorageKind,
    TensorType,
)
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.layout_algebra import cosize
from tilefoundry.ir.types.mesh import levels, make_mesh, starts
from tilefoundry.ir.visitor import StmtVisitor, collect_exprs
from tilefoundry.schedule import finalize
from tilefoundry.visitor_registry.contexts import VerifyContext
from tilefoundry.visitor_registry.instruction import check_call


def _region_key(region: MeshRegion, occurrences: dict[str, int]) -> str:
    """Name one region by its selected topology run and lexical occurrence."""
    mesh = region.mesh
    assert len(mesh.topologies) == 1
    topology = getattr(mesh.topologies[0], "name", mesh.topologies[0])
    offset = starts(mesh)[0]
    size = prod(flatten(levels(mesh)[0].shape))
    selection = f"{topology}@{offset}:{size}"
    occurrence = occurrences.get(selection, 0)
    occurrences[selection] = occurrence + 1
    return f"{selection}#{occurrence}"


class _OperandMatchVisitor(StmtVisitor[None]):
    """Exercise operand declarations directly, independently of the report renderer."""

    def __init__(self) -> None:
        self.matches: list[tuple[object, str, dict]] = []

    def visit_MeshScope(self, stmt: MeshScope) -> None:
        previous = getattr(self, "meshes", ())
        self.meshes = (*previous, stmt.mesh)
        try:
            self.visit(stmt.body)
        finally:
            self.meshes = previous

    def visit_Evaluate(self, stmt: Evaluate) -> None:
        op = stmt.callable
        if getattr(type(op), "_op_schema", None) is None:
            return
        meshes = getattr(self, "meshes", ())
        mesh = make_mesh(*meshes) if meshes else None
        bindings, projection = check_call(
            op,
            stmt.args,
            mesh,
            frame=VerifyContext(current_mesh=mesh, mesh_scope=meshes),
        )
        for param, pattern in zip(projection.params, projection.patterns, strict=True):
            if pattern is not None:
                self.matches.append((op, param.name, dict(bindings)))


def _direct_operand_matches(function: PrimFunction) -> list[tuple[object, str, dict]]:
    visitor = _OperandMatchVisitor()
    visitor.visit(function.body)
    return visitor.matches


def _assert_live_smem_views_do_not_overlap(function: PrimFunction, source: Path) -> None:
    """Check literal views in lowered TIR, including uses across loop backedges."""
    views: list[tuple[Var, int, int]] = []
    defined: dict[int, int] = {}
    last: dict[int, int] = {}
    aliases: dict[int, set[int]] = {}
    clock = 0

    def use(expressions) -> None:
        for expression in expressions:
            for value in collect_exprs(expression):
                if isinstance(value, Var):
                    for key in aliases.get(id(value), {id(value)}):
                        if key in defined:
                            last[key] = clock

    def walk(statement) -> None:
        nonlocal clock
        if statement is None:
            return
        clock += 1
        if isinstance(statement, Sequential):
            for child in statement.body:
                walk(child)
        elif isinstance(statement, LetStmt):
            use((statement.value,))
            defined[id(statement.var)] = last[id(statement.var)] = clock
            aliases[id(statement.var)] = {
                key
                for value in collect_exprs(statement.value)
                if isinstance(value, Var)
                for key in aliases.get(id(value), {id(value)})
            } or {id(statement.var)}
            fields = (
                statement.value.elements
                if isinstance(statement.value, Tuple)
                else (statement.value,)
            )
            for field in fields:
                if (
                    isinstance(field, Call)
                    and isinstance(field.target, TensorView)
                    and field.target.storage is StorageKind.SMEM
                    and isinstance(field.args[0], Constant)
                ):
                    span = (
                        cosize(field.type.layout)
                        if field.type.layout is not None
                        else prod(field.type.shape)
                    )
                    views.append(
                        (
                            statement.var,
                            int(field.args[0].value),
                            span * field.type.dtype.bit_width // 8,
                        )
                    )
        elif isinstance(statement, (For, While)):
            use(
                (statement.start, statement.stop, statement.step)
                if isinstance(statement, For)
                else (statement.cond,)
            )
            entered = clock
            walk(statement.body)
            for key, used in last.items():
                if defined[key] < entered < used:
                    last[key] = clock
        elif isinstance(statement, If):
            use((statement.cond,))
            walk(statement.then_body)
            walk(statement.else_body)
        elif isinstance(statement, MeshScope):
            walk(statement.body)
        elif isinstance(statement, Evaluate):
            use(statement.args)

    walk(function.body)
    for (left, start, size), (right, other_start, other_size) in combinations(views, 2):
        if left is right:
            continue
        live = max(defined[id(left)], defined[id(right)]) <= min(last[id(left)], last[id(right)])
        overlap = max(start, other_start) < min(start + size, other_start + other_size)
        assert not (live and overlap), (
            f"{source.name}: live smem views overlap: "
            f"{left.name}@{start}+{size}, {right.name}@{other_start}+{other_size}"
        )


class _RetainedStatements(StmtVisitor[None]):
    def __init__(self):
        self.calls = []
        self.loops = []
        self.operations = []
        self.statements = []
        self.bindings = {}

    def visit_LetStmt(self, stmt, ctx=None):
        self.bindings[id(stmt.var)] = stmt.value

    def storage(self, value):
        if isinstance(value, Var) and id(value) in self.bindings:
            return self.storage(self.bindings[id(value)])
        if isinstance(value, Call) and isinstance(value.target, (TensorView, PtrOf, Slice)):
            return self.storage(value.args[0])
        return value

    def visit_For(self, stmt, ctx=None):
        self.loops.append(stmt)
        self.visit(stmt.body)

    def visit_Evaluate(self, stmt, ctx=None):
        self.operations.append(stmt.callable)
        self.statements.append(stmt)
        if isinstance(stmt.callable, (PrimFunction, ModuleList, Call)):
            self.calls.append(stmt)


def _finalized(module):
    authored = module.entry_function()
    result = analyze(*check_program(module, authored), analysis=("memory",))
    return finalize(result)


def _copy_schedule_call(
    *, repeat=None, order=None, storage: StorageKind = StorageKind.GMEM
) -> Call:
    layout = Layout((4,), (1,))
    type_ = TensorType((4,), DType.bf16, layout, storage)
    return Call(
        target=ScheduleOp(
            op=CopyAsync(),
            repeat=repeat,
            order=order,
        ),
        args=(
            Var(name="src", type=type_),
            Call(
                Empty(type=replace(type_, storage=StorageKind.SMEM)),
                (),
                type=replace(type_, storage=StorageKind.SMEM),
            ),
        ),
        type=type_,
    )


def _evaluate_call(
    op: Op,
    typed_inputs: tuple[tuple[TensorType, torch.Tensor], ...],
    result_type: TensorType,
) -> torch.Tensor:
    params = tuple(
        Var(name=f"arg{index}", type=type_) for index, (type_, _data) in enumerate(typed_inputs)
    )
    call = Call(target=op, args=params, type=result_type)
    function = Function.build(
        name="schedule_value",
        params=params,
        body=call,
        return_type=result_type,
    )
    return evaluate(function, *(data for _type, data in typed_inputs))


@register_op(dialect="T", category="schedule", name="unstated_instruction")
class _UnstatedInstruction(Op):
    execution_mesh = thread_execution_mesh()

    value = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=is_ranked_tensor(),
    )


def entry_function(program):
    return program.entry_function() if isinstance(program, Module) else program
