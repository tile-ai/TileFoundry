"""Instruction candidates for unscheduled HIR calls."""

from __future__ import annotations

from typing import Any, Mapping

from tilefoundry.analysis.check import check_program
from tilefoundry.inspection import PatternPrinter, PythonPrinter
from tilefoundry.ir.core import Call, OpCapability, get_metadata, op_identifier, value_label
from tilefoundry.ir.core.metadata import SourceSpanMetadata
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.schedule import ScheduleRefusal, match_instruction
from tilefoundry.ir.types import StorageKind, TensorType
from tilefoundry.ir.types.dim import is_dim_op_call
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.ir.visitor import collect_exprs, expr_children
from tilefoundry.schedule._reporting import capability_families
from tilefoundry.target import Target
from tilefoundry.visitor_registry.access_relation import access_relation_registry
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.call_frame import CallFrame, CallFrameVisitor
from tilefoundry.visitor_registry.candidates import (
    candidate_operands,
    candidate_ops,
    sole_candidate,
)


def _instructions(target: Target) -> tuple[tuple[type, OpCapability], ...]:
    """Return discoverable declarations that state comparable coordinates."""
    return tuple(
        (op_type, capability)
        for op_type, capabilities in capability_families(target)
        if access_relation_registry.lookup(op_type) is not None
        for capability in capabilities
    )


def candidates(
    module, entry, *, source: str | None = None, dims: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Report instructions using the same acceptance boundary as explicit schedule inference."""
    module, entry = check_program(module, entry, dims=dims)
    target = module.resolve_target()
    owners = {}
    visited = set()

    def enter(frame, outer_mesh=None):
        if id(frame.function) in visited:
            return
        visited.add(id(frame.function))
        seen = set()

        def visit(expr, mesh):
            if id(expr) in seen:
                return
            seen.add(id(expr))
            if isinstance(expr, (MeshRegion, LoopRegion)):
                for argument in expr.args:
                    visit(argument, mesh)
                inner = mesh
                if isinstance(expr, MeshRegion):
                    inner = make_mesh(mesh, expr.mesh) if mesh else expr.mesh
                visit(expr.body, inner)
                if isinstance(expr, LoopRegion):
                    for value in expr.yield_values:
                        visit(value, inner)
            else:
                for child in expr_children(expr):
                    visit(child, mesh)
            if isinstance(expr, Call):
                owners[id(expr)] = frame, mesh
                callee = CallFrameVisitor.callee_frame(expr, frame)
                if callee is not None:
                    enter(callee, mesh)

        visit(frame.function.body, outer_mesh)

    enter(CallFrame(module, entry))
    declared = _instructions(target)
    type_printer = PythonPrinter()
    pattern_printer = PatternPrinter()
    rows = []
    calls = []
    frames = CallFrameVisitor.walk_frames(CallFrame(module, entry))
    for call in (call for frame in frames for call in collect_exprs(frame.function.body)):
        if (
            not isinstance(call, Call)
            or is_dim_op_call(call)
            or not isinstance(call.type, TensorType)
        ):
            continue
        instructions = candidate_ops(type(call.target))
        if not instructions or aliased_operand(call) is not None:
            continue
        if call.type.storage is StorageKind.UMAT and call.type.shape == ():
            continue
        calls.append(call)
        frame, mesh = owners[id(call)]
        level = None if mesh is None else mesh.topologies[-1].name
        usable, refused = [], []
        for op_type, capability in declared:
            if op_type not in instructions:
                continue
            identifier = op_identifier(capability.declaration or op_type)
            try:
                attributes, types = candidate_operands(call, op_type)
                if capability.attribute is not None:
                    attributes[capability.attribute] = capability.declaration
                instances = match_instruction(op_type, attributes, types, mesh, frame=frame)
            except ScheduleRefusal as error:
                refused.append(
                    {"id": identifier, **pattern_printer.match_data((error.path,), error.reason)}
                )
                continue
            candidate = {
                "id": identifier,
                **pattern_printer.match_data(tuple(match.path for match in instances)),
            }
            if sole_candidate(call) is not None:
                candidate["default"] = True
            usable.append(candidate)
        parameters = [
            param for param in type(call.target)._op_schema.signature if param.kind == "input"
        ]
        operands = [
            f"{param.name}={type_printer.print(arg.type)}"
            for param, arg in zip(parameters, call.args, strict=True)
        ]
        operands.append(f"result={type_printer.print(call.type)}")
        rows.append(
            {
                "line": value_label(call) or type(call.target)._op_schema.name,
                "op": op_identifier(type(call.target)),
                "scope": "whole program" if level is None else f"per {level}",
                "operands": operands,
                "candidates": usable,
                "refused": refused,
            }
        )
    if not calls:
        raise ValueError("source has no unscheduled candidate site")
    if source is None:
        source = next(
            (
                span.file
                for call in calls
                if (span := get_metadata(call, SourceSpanMetadata)) is not None
            ),
            module.name,
        )
    return {"source": source, "target": target.identity, "lines": rows}


def render(data: dict[str, Any]) -> str:
    """Render one candidate report as stable text."""
    lines = [f"source {data['source']}", f"target {data['target']}"]
    for row in data["lines"]:
        lines.append(f"  {row['line']}  {row['op']}  {row['scope']}  {'  '.join(row['operands'])}")
        for fit in row["candidates"]:
            kind = "default" if fit.get("default", False) else "candidate"
            written = PatternPrinter().match_text(fit["id"], fit)
            lines.append(f"    {kind:<12}{written}")
        for rejection in row["refused"]:
            written = PatternPrinter().match_text(rejection["id"], rejection).splitlines()
            lines.append(f"    refused     {written[0]}")
            lines.extend(f"                {line}" for line in written[1:])
    return "\n".join(lines)


__all__ = ["candidates", "render"]
