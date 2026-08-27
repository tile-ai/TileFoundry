"""Build target-aware analysis report data without inspection rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, fields, is_dataclass
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

from tilefoundry.analysis.facts import MemoryHierarchyFacts
from tilefoundry.analysis.memory import cache_pressure
from tilefoundry.analysis.metadata import (
    ComputeCostMetadata,
    LoopFootprintMetadata,
    MemoryMetadata,
    PerformanceMetadata,
    PerformanceSummaryMetadata,
    RooflineMetadata,
    TrafficMetadata,
)
from tilefoundry.analysis.walk import collect_exprs, tensor_types
from tilefoundry.ir.core import Call, IRMetadata, binding_name, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr

_FAMILIES: dict[type[IRMetadata], str] = {}
_EXPR_FIELDS: dict[type[IRMetadata], dict[str, Callable[..., object]]] = {}


def _derived_family(record: type) -> str:
    """The family name the record class already states."""
    stem = record.__name__.removesuffix("Metadata")
    return re.sub(r"(?<!^)(?=[A-Z])", "-", stem).lower()


def declare_record(record: type[IRMetadata], *, family: str | None = None) -> None:
    """Declare one structured record and its report family."""
    _FAMILIES[record] = family or _derived_family(record)


def declared_records() -> tuple[type[IRMetadata], ...]:
    """Every record type declared as structured report data."""
    return tuple(_FAMILIES)


def family_of(record: type[IRMetadata]) -> str:
    """The report family that owns *record*."""
    return _FAMILIES.get(record, _derived_family(record))


def expr_field(
    record: type[IRMetadata], key: str, of: Callable[..., object]
) -> None:
    """Read one report field from the expression carrying its record."""
    _EXPR_FIELDS.setdefault(record, {})[key] = of


def expr_fields(record: type[IRMetadata]) -> frozenset[str]:
    """Which fields of *record* are read from its expression."""
    return frozenset(_EXPR_FIELDS.get(record, {}))


def render_record(record: IRMetadata, expr: object) -> dict[str, object]:
    """Serialize every record field under its declared field name."""
    from_expr = _EXPR_FIELDS.get(type(record), {})
    hints = get_type_hints(type(record))
    reported: dict[str, object] = {}
    for name in (item.name for item in fields(record)):
        if name in from_expr:
            value = from_expr[name](record, expr)
            if value is not None:
                reported[name] = value
            continue
        reported[name] = _reported_value(getattr(record, name), hints[name])
    return reported


def _reported_value(value: object, declared: object) -> object:
    """Serialize one value according to the type its field declared.

    A field that may be absent declares a union with ``None``; the value that
    arrived is not ``None`` here, so what remains of the union is what it is.
    """
    if value is None:
        return None
    if get_origin(declared) in (Union, UnionType):
        stated = [arg for arg in get_args(declared) if arg is not type(None)]
        if len(stated) == 1:
            declared = stated[0]
    if is_dataclass(declared) and isinstance(declared, type):
        hints = get_type_hints(declared)
        return {
            item.name: _reported_value(getattr(value, item.name), hints[item.name])
            for item in fields(declared)
        }
    origin = get_origin(declared)
    if origin is dict:
        _, value_type = get_args(declared)
        return {key: _reported_value(item, value_type) for key, item in value.items()}
    if origin in (tuple, list):
        item_type = _item_type(declared)
        pair = _pair_types(item_type)
        if pair is not None:
            return {key: _reported_value(item, pair[1]) for key, item in value}
        return [_reported_value(item, item_type) for item in value]
    return value


def _item_type(declared: object) -> object:
    args = [arg for arg in get_args(declared) if arg is not Ellipsis]
    return args[0] if args else object


def _pair_types(declared: object) -> tuple[object, object] | None:
    if get_origin(declared) is not tuple:
        return None
    args = get_args(declared)
    if len(args) != 2 or Ellipsis in args:
        return None
    return args[0], args[1]


for _record_type in (
    ComputeCostMetadata,
    LoopFootprintMetadata,
    MemoryMetadata,
    RooflineMetadata,
):
    declare_record(_record_type)

declare_record(PerformanceMetadata, family="performance")
declare_record(PerformanceSummaryMetadata, family="performance")


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
    record: TrafficMetadata, expr: object
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


expr_field(TrafficMetadata, "operands", _operands)


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
    asked = {ComputeCostMetadata, TrafficMetadata}
    if asked & selected or (
        "roofline" in analyses and asked & available
    ):
        data["totals"] = _work_totals(function)
    return data


def _call_records(
    function: Function,
    selected: frozenset[type[IRMetadata]],
    call_labels: Mapping[int, str] | None,
) -> list[dict[str, object]]:
    """Every selected record attached to a Call, in program order."""
    rows: list[dict[str, object]] = []
    for expr in collect_exprs(function.body):
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
    for expr in collect_exprs(function.body):
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
    """The multiplicity-aware work and movement recorded on the Function root.

    Two families answer here, each about its own half, so a report states what
    a program asks of the machine beside what it moves through it.
    """
    record = get_metadata(function, ComputeCostMetadata)
    moved = get_metadata(function, TrafficMetadata)
    return {
        "flops": {} if record is None else render_record(record, function)["flops"],
        "traffic": {} if moved is None else render_record(moved, function)["whole"],
    }


def render_json(data: dict[str, object]) -> str:
    """Render report data as sorted JSON for byte-stable comparisons."""
    return json.dumps(data, indent=2, sort_keys=True)


__all__ = [
    "declare_record",
    "declared_records",
    "expr_field",
    "expr_fields",
    "family_of",
    "render_json",
    "render_record",
    "report_data",
    "selected_types",
]
