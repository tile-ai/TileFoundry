"""Instruction candidates for unscheduled HIR calls."""

from __future__ import annotations

from typing import Any

from tilefoundry import analysis, inspection, target
from tilefoundry.interfaces import access, schedule, typeinfer, valuation
from tilefoundry.ir import core, expr, hir, types

from .reporting import capability_families


def discoverable_instructions(tgt: target.Target) -> tuple[tuple[type, expr.OpCapability], ...]:
    """Return discoverable declarations that state comparable coordinates."""
    return tuple(
        (op_type, capability)
        for op_type, capabilities in capability_families(tgt)
        if access.relation_registry.has(op_type)
        for capability in capabilities
    )


def candidates(
    module, entry, *, source: str | None = None
) -> dict[str, Any]:
    """Report instructions for a program ``check_program`` returned."""
    tgt = module.resolve_target()
    contexts = {}
    frames = tuple(valuation.definitions(valuation.Context(module, entry)))
    for frame in frames:
        mesh = expr.get_metadata(frame.function, analysis.ExecutionMeshMetadata)
        contexts.update(
            valuation.index_contexts(
                frame.function,
                typeinfer.TypeInferContext(
                    module=frame.module, function=frame.function, target=frame.target, topologies=frame.topologies,
                    owner=frame.function,
                    current_mesh=None if mesh is None else mesh.mesh,
                ),
            )
        )
    declared = discoverable_instructions(tgt)
    type_printer = inspection.PythonPrinter()
    pattern_printer = inspection.PatternPrinter()
    rows = []
    calls = []
    for call in (call for frame in frames for call in core.collect_exprs(frame.function.body)):
        if (
            not isinstance(call, expr.Call)
            or expr.is_dim_op_call(call)
            or not isinstance(call.type, types.TensorType)
        ):
            continue
        instructions = schedule.candidate_ops(type(call.target), call=call)
        if not instructions or access.aliased_operand(call) is not None:
            continue
        if call.type.storage is expr.StorageKind.UMAT and call.type.shape == ():
            continue
        calls.append(call)
        context = contexts[id(call)]
        frame, mesh = context, context.current_mesh
        level = None if mesh is None else mesh.topologies[-1].name
        usable, refused = [], []
        for op_type, capability in declared:
            if op_type not in instructions:
                continue
            identifier = expr.get_registered_name(capability.declaration or op_type)
            try:
                instances = []
                rejection = None
                for attributes, type_list in schedule.candidate_bindings(call, op_type):
                    if capability.attribute is not None:
                        attributes[capability.attribute] = capability.declaration
                    try:
                        instances.extend(hir.schedule.match_instruction(op_type, attributes, type_list, mesh, frame=frame))
                    except (schedule.RefusalError, ValueError, TypeError) as error:
                        rejection = error
                if not instances:
                    raise rejection if isinstance(rejection, schedule.RefusalError) else schedule.RefusalError(
                        str(rejection) if rejection is not None else "no applicable HIR declaration"
                    )
            except schedule.RefusalError as error:
                refused.append(
                    {"id": identifier, **pattern_printer.match_data((error.path,), error.reason)}
                )
                continue
            candidate = {
                "id": identifier,
                **pattern_printer.match_data(tuple(match.path for match in instances)),
            }
            if schedule.sole_candidate(call, target=tgt) is not None:
                candidate["default"] = True
            usable.append(candidate)
        parameters = [
            param for param in expr.schema_of(type(call.target)).signature if param.kind == "input"
        ]
        operands = [
            f"{param.name}={type_printer.print(arg.type)}"
            for param, arg in zip(parameters, call.args, strict=True)
        ]
        operands.append(f"result={type_printer.print(call.type)}")
        rows.append(
            {
                "line": expr.value_label(call) or expr.schema_of(type(call.target)).name,
                "op": expr.get_registered_name(call.target),
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
                if (span := expr.get_metadata(call, expr.SourceSpanMetadata)) is not None
            ),
            module.name,
        )
    return {"source": source, "target": tgt.identity, "lines": rows}


def render(data: dict[str, Any]) -> str:
    """Render one candidate report as stable text."""
    lines = [f"source {data['source']}", f"target {data['target']}"]
    for row in data["lines"]:
        lines.append(f"  {row['line']}  {row['op']}  {row['scope']}  {'  '.join(row['operands'])}")
        for fit in row["candidates"]:
            kind = "default" if fit.get("default", False) else "candidate"
            written = inspection.PatternPrinter().match_text(fit["id"], fit)
            lines.append(f"    {kind:<12}{written}")
        for rejection in row["refused"]:
            written = inspection.PatternPrinter().match_text(rejection["id"], rejection).splitlines()
            lines.append(f"    refused     {written[0]}")
            lines.extend(f"                {line}" for line in written[1:])
    return "\n".join(lines)


__all__ = ["candidates", "render"]
