"""Shared scheduling assertions and unit-test program builders."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from math import prod
from pathlib import Path

import torch

from tilefoundry import analysis
from tilefoundry.interfaces import evaluator, schedule, verify
from tilefoundry.ir import core, expr, hir, pattern, tir, types
from tilefoundry.ir.hir import tensor
from tilefoundry.ir.tir import memory
from tilefoundry.ir.types import layout_algebra
from tilefoundry.schedule import finalize


def _region_key(region: core.MeshRegion, occurrences: dict[str, int]) -> str:
    """Name one region by its selected topology run and lexical occurrence."""
    mesh = region.mesh
    assert len(mesh.topologies) == 1
    topology = getattr(mesh.topologies[0], "name", mesh.topologies[0])
    offset = types.starts(mesh)[0]
    size = prod(types.flatten(types.levels(mesh)[0].shape))
    selection = f"{topology}@{offset}:{size}"
    occurrence = occurrences.get(selection, 0)
    occurrences[selection] = occurrence + 1
    return f"{selection}#{occurrence}"


class _OperandMatchVisitor(core.StmtVisitor[None]):
    """Exercise operand declarations directly, independently of the report renderer."""

    def __init__(self) -> None:
        self.matches: list[tuple[object, str, dict]] = []

    def visit_MeshScope(self, stmt: core.MeshScope) -> None:
        previous = getattr(self, "meshes", ())
        self.meshes = (*previous, stmt.mesh)
        try:
            self.visit(stmt.body)
        finally:
            self.meshes = previous

    def visit_Evaluate(self, stmt: core.Evaluate) -> None:
        op = stmt.callable
        if expr.schema_of(type(op)) is None:
            return
        meshes = getattr(self, "meshes", ())
        mesh = types.Mesh.from_meshes(*meshes) if meshes else None
        bindings, projection = schedule.check_call(
            op,
            stmt.args,
            mesh,
            frame=verify.Context(current_mesh=mesh, mesh_scope=meshes),
        )
        for param, declared in zip(projection.params, projection.patterns, strict=True):
            if declared is not None:
                self.matches.append((op, param.name, dict(bindings)))


def _direct_operand_matches(function: core.PrimFunction) -> list[tuple[object, str, dict]]:
    visitor = _OperandMatchVisitor()
    visitor.visit(function.body)
    return visitor.matches


def _assert_live_smem_views_do_not_overlap(function: core.PrimFunction, source: Path) -> None:
    """Check literal views in lowered TIR, including uses across loop backedges."""
    views: list[tuple[expr.Var, int, int]] = []
    defined: dict[int, int] = {}
    last: dict[int, int] = {}
    aliases: dict[int, set[int]] = {}
    clock = 0

    def use(expressions) -> None:
        for expression in expressions:
            for value in core.collect_exprs(expression):
                if isinstance(value, expr.Var):
                    for key in aliases.get(id(value), {id(value)}):
                        if key in defined:
                            last[key] = clock

    def walk(statement) -> None:
        nonlocal clock
        if statement is None:
            return
        clock += 1
        if isinstance(statement, core.Sequential):
            for child in statement.body:
                walk(child)
        elif isinstance(statement, core.LetStmt):
            use((statement.value,))
            defined[id(statement.var)] = last[id(statement.var)] = clock
            aliases[id(statement.var)] = {
                key
                for value in core.collect_exprs(statement.value)
                if isinstance(value, expr.Var)
                for key in aliases.get(id(value), {id(value)})
            } or {id(statement.var)}
            fields = (
                statement.value.elements
                if isinstance(statement.value, expr.Tuple)
                else (statement.value,)
            )
            for field in fields:
                if (
                    isinstance(field, expr.Call)
                    and isinstance(field.target, memory.TensorView)
                    and field.target.storage is expr.StorageKind.SMEM
                    and isinstance(field.args[0], expr.Constant)
                ):
                    span = (
                        layout_algebra.cosize(field.type.layout)
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
        elif isinstance(statement, (core.For, core.While)):
            use(
                (statement.start, statement.stop, statement.step)
                if isinstance(statement, core.For)
                else (statement.cond,)
            )
            entered = clock
            walk(statement.body)
            for key, used in last.items():
                if defined[key] < entered < used:
                    last[key] = clock
        elif isinstance(statement, core.If):
            use((statement.cond,))
            walk(statement.then_body)
            walk(statement.else_body)
        elif isinstance(statement, core.MeshScope):
            walk(statement.body)
        elif isinstance(statement, core.Evaluate):
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


class _RetainedStatements(core.StmtVisitor[None]):
    def __init__(self):
        self.calls = []
        self.loops = []
        self.operations = []
        self.statements = []
        self.bindings = {}

    def visit_LetStmt(self, stmt, ctx=None):
        self.bindings[id(stmt.var)] = stmt.value

    def storage(self, value):
        if isinstance(value, expr.Var) and id(value) in self.bindings:
            return self.storage(self.bindings[id(value)])
        if isinstance(value, expr.Call) and isinstance(value.target, (memory.TensorView, memory.PtrOf, tensor.Slice)):
            return self.storage(value.args[0])
        return value

    def visit_For(self, stmt, ctx=None):
        self.loops.append(stmt)
        self.visit(stmt.body)

    def visit_Evaluate(self, stmt, ctx=None):
        self.operations.append(stmt.callable)
        self.statements.append(stmt)
        if isinstance(stmt.callable, (core.PrimFunction, core.FunctionRef)):
            self.calls.append(stmt)


def _finalized(module):
    authored = module.entry_function()
    result = analysis.analyze(*analysis.check_program(module, authored), analysis=("memory", "sync"))
    return finalize(result)


def _copy_schedule_call(
    *, repeat=None, order=None, storage: expr.StorageKind = expr.StorageKind.GMEM
) -> expr.Call:
    layout = types.Layout((4,), (1,))
    type_ = types.TensorType((4,), expr.DType.bf16, layout, storage)
    return expr.Call(
        target=hir.schedule.ScheduleOp(
            op=tir.async_copy.CopyAsync(),
            repeat=repeat,
            order=order,
        ),
        args=(
            expr.Var(name="src", type=type_),
            expr.Call(
                tensor.Empty(type=replace(type_, storage=expr.StorageKind.SMEM)),
                (),
                type=replace(type_, storage=expr.StorageKind.SMEM),
            ),
        ),
        type=type_,
    )


def _evaluate_call(
    op: expr.Op,
    typed_inputs: tuple[tuple[types.TensorType, torch.Tensor], ...],
    result_type: types.TensorType,
) -> torch.Tensor:
    params = tuple(
        expr.Var(name=f"arg{index}", type=type_) for index, (type_, _data) in enumerate(typed_inputs)
    )
    call = expr.Call(target=op, args=params, type=result_type)
    function = core.Function.from_parts(
        name="schedule_value",
        params=params,
        body=call,
        return_type=result_type,
    )
    return evaluator.evaluate(function, *(data for _type, data in typed_inputs))


@expr.register_op(dialect="T", category="schedule", name="unstated_instruction")
class _UnstatedInstruction(expr.Op):
    execution_mesh = pattern.thread_execution_mesh()

    value = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ | expr.MemoryEffect.WRITE,
        pattern=pattern.is_ranked_tensor(),
    )


def entry_function(program):
    return program.entry_function() if isinstance(program, core.Module) else program
