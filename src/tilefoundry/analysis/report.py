"""Build target-aware analysis report data without inspection rendering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict

from tilefoundry.analysis.facts import MemoryHierarchyFacts
from tilefoundry.analysis.memory import cache_pressure
from tilefoundry.analysis.metadata import (
    ComputeCostMetadata,
    LoopFootprintMetadata,
    MemoryMetadata,
)
from tilefoundry.analysis.walk import postorder, tensor_types
from tilefoundry.inspection.values import (
    declared_records,
    expr_field,
    family_of,
    render_record,
)
from tilefoundry.ir.core import Call, IRMetadata, binding_name, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr


def _type_text(type_: object) -> str:
    """One operand's type, short enough to sit on a report line."""
    tensors = tensor_types(type_)
    if not tensors:
        return str(type_)
    first = tensors[0]
    shape = ",".join(str(dim) for dim in first.shape)
    more = "+" if len(tensors) > 1 else ""
    storage = f" {first.storage}"
    return f"{first.dtype.name}[{shape}]{more}{storage}"


def _operand_name(operand: object) -> str:
    """What to call this operand, from the program rather than from the record."""
    name = binding_name(operand) or getattr(operand, "name", None)
    if name:
        return name
    if isinstance(operand, Call):
        return type(operand.target).__name__.lower()
    return type(operand).__name__.lower()


def _operands(
    record: ComputeCostMetadata, expr: object
) -> list[dict[str, object]] | None:
    """Each recorded amount, against the operand it was charged to."""
    if not isinstance(expr, Call) or not record.operands:
        return None
    operands = (*expr.args, expr)
    return [
        {
            "arg": "result" if index == len(expr.args) else index,
            "name": _operand_name(operand),
            "type": _type_text(operand.type),
            "read": moved.read,
            "write": moved.write,
        }
        for index, (operand, moved) in enumerate(zip(operands, record.operands))
    ]


expr_field(ComputeCostMetadata, "operands", _operands)


def _records_of(expr: object, selected: frozenset[type[IRMetadata]]) -> dict[str, object]:
    """Every selected record on *expr*, keyed by the family that owns it."""
    result: dict[str, object] = {}
    for metadata_type in declared_records():
        if metadata_type not in selected:
            continue
        record = get_metadata(expr, metadata_type)
        if record is not None:
            result[family_of(metadata_type)] = render_record(record, expr)
    return result


def selected_types(
    module: Module,
    analyses: tuple[str, ...],
    metadata_types: tuple[type[IRMetadata], ...],
) -> tuple[type[IRMetadata], ...]:
    """Return record types requested analyses own and actually wrote."""
    target = module.resolve_target()
    owned: set[type[IRMetadata]] = set()
    for analysis in analyses:
        owned.update(target.get_analyzer(analysis).produces)
    order: list[type[IRMetadata]] = []
    for metadata_type in metadata_types:
        if metadata_type in owned and metadata_type not in order:
            order.append(metadata_type)
    return tuple(order)


def report_data(
    *,
    module: Module,
    function: Function,
    analyses: tuple[str, ...],
    level: str | None,
    executed: tuple[str, ...],
    metadata_types: tuple[type[IRMetadata], ...],
    call_labels: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Project one completed analysis into stable, serializable report data."""
    selected_types_ = selected_types(module, analyses, metadata_types)
    selected = frozenset(selected_types_)
    target = module.resolve_target()
    function_records = _records_of(function, selected)
    if "roofline" in analyses and "memory" not in function_records:
        support = _roofline_memory_support(function)
        if support is not None:
            function_records["memory"] = support
    data = {
        "target": target.identity,
        "module": module.name,
        "function": function.name,
        "topology": level,
        "requested": list(analyses),
        "executed": list(executed),
        "function_records": function_records,
        "calls": _call_records(function, selected, call_labels),
        "loops": _loop_records(function, selected, target),
    }
    available = set(metadata_types)
    if ComputeCostMetadata in selected or (
        "roofline" in analyses and ComputeCostMetadata in available
    ):
        data["totals"] = _work_totals(function)
    return data


def _roofline_memory_support(function: Function) -> dict[str, object] | None:
    """The bounded memory fact that explains a requested roofline verdict."""
    record = get_metadata(function, MemoryMetadata)
    if record is None:
        return None
    return {
        "footprint": [
            {"level": value.level, "peak_bytes": value.peak_bytes}
            for value in record.footprint
        ]
    }


def _call_records(
    function: Function,
    selected: frozenset[type[IRMetadata]],
    call_labels: Mapping[int, str] | None,
) -> list[dict[str, object]]:
    """Every selected record attached to a Call, in program order."""
    rows: list[dict[str, object]] = []
    for expr in postorder(function.body):
        if not isinstance(expr, Call):
            continue
        records = _records_of(expr, selected)
        if not records:
            continue
        if call_labels is not None:
            value = call_labels.get(id(expr))
            if value is None:
                continue
        else:
            value = binding_name(expr) or type(expr.target).__name__.lower()
        rows.append({"value": value, **records})
    return rows


def _loop_records(
    function: Function,
    selected: frozenset[type[IRMetadata]],
    target,
) -> list[dict[str, object]]:
    """Every selected record attached to an authored loop."""
    memory = get_metadata(function, MemoryMetadata)
    facts = (
        target.get_facts(MemoryHierarchyFacts)
        if memory is not None and MemoryMetadata in selected
        else None
    )
    peaks = (
        {item.level: item.peak_bytes for item in memory.footprint}
        if memory is not None
        else {}
    )
    rows: list[dict[str, object]] = []
    for expr in postorder(function.body):
        if not isinstance(expr, GridRegionExpr):
            continue
        records = _records_of(expr, selected)
        if not records:
            continue
        record = get_metadata(expr, LoopFootprintMetadata)
        if record is not None and facts is not None:
            pressure = cache_pressure(record, facts, peaks)
            if pressure:
                records["cache-pressure"] = [asdict(item) for item in pressure]
        rows.append({"value": expr.induction_var.name, **records})
    return rows


def _work_totals(function: Function) -> dict[str, object]:
    """The multiplicity-aware work recorded on the Function root."""
    record = get_metadata(function, ComputeCostMetadata)
    if record is None:
        return {"flops": {}, "traffic": {}}
    reported = render_record(record, function)
    return {"flops": reported["flops"], "traffic": reported["traffic"]}


def render_json(data: dict[str, object]) -> str:
    """Render report data as sorted JSON for byte-stable comparisons."""
    return json.dumps(data, indent=2, sort_keys=True)


__all__ = ["render_json", "report_data", "selected_types"]
