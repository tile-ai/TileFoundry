"""Report data of an analysis result: the records on the IR, projected for rendering."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from types import MappingProxyType, UnionType
from typing import Union, get_args, get_origin, get_type_hints

from tilefoundry import analysis
from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr, utils

from .python_printer import HirPrinter


def render_record(record: expr.IRMetadata, node: object, *, printer=None) -> dict[str, object]:
    """Serialize every record field under its declared field name."""
    if printer is None:
        printer = HirPrinter()
    from_expr = EXPR_FIELDS.get(type(record), {})
    hints = get_type_hints(type(record))
    reported: dict[str, object] = {}
    for name in (item.name for item in fields(record)):
        if name in from_expr:
            value = from_expr[name](record, node, printer=printer)
            if value is not None:
                reported[name] = value
            continue
        reported[name] = reported_value(getattr(record, name), hints[name], printer)
    return reported


def reported_value(value: object, declared: object, printer=None) -> object:
    """Serialize one value according to the type its field declared.

    A field that may be absent declares a union with ``None``; the value that
    arrived is not ``None`` here, so what remains of the union is what it is.
    """
    if value is None:
        return None
    if isinstance(value, core.SSAValue):
        return printer.print(value)
    if isinstance(value, Enum):
        return value.value
    if get_origin(declared) in (Union, UnionType):
        stated = [arg for arg in get_args(declared) if arg is not type(None)]
        if len(stated) == 1:
            declared = stated[0]
    if isinstance(value, analysis.Footprint):
        return {
            "buffers": {
                "/".join((*ownership, printer.print(buffer))): reported_value(held, analysis.Breakdown[int], printer)
                for buffer, ownership, held in value.buffers
            },
            "precision": value.precision.value,
        }
    if is_dataclass(declared) and isinstance(declared, type):
        hints = get_type_hints(declared)
        result = {
            item.name: reported_value(getattr(value, item.name), hints[item.name], printer)
            for item in fields(declared)
        }
        if isinstance(value, analysis.ValueLifetime):
            result["binding"] = result.pop("value")
        elif isinstance(value, analysis.ReuseWindow):
            result["buffer"] = "/".join((*result.pop("ownership"), result["buffer"]))
        return result
    origin = get_origin(declared)
    if origin is analysis.Breakdown:
        (item,) = get_args(declared)
        return {kind: reported_value(spread, analysis.Spread[item], printer) for kind, spread in value.kinds}
    if origin is analysis.Spread:
        (item,) = get_args(declared)
        return {
            "logical": reported_value(value.logical, item, printer),
            "total": reported_value(value.total, item, printer),
            "per_unit": [reported_value(share, item, printer) for share in value.per_unit],
        }
    if origin is dict:
        _, value_type = get_args(declared)
        return {key: reported_value(item, value_type, printer) for key, item in value.items()}
    if origin in (tuple, list):
        item_type = element_type(declared)
        pair = pair_types(item_type)
        if pair is not None:
            return {key: reported_value(item, pair[1], printer) for key, item in value}
        return [reported_value(item, item_type, printer) for item in value]
    return value


def element_type(declared: object) -> object:
    args = [arg for arg in get_args(declared) if arg is not Ellipsis]
    return args[0] if args else object


def pair_types(declared: object) -> tuple[object, object] | None:
    if get_origin(declared) is not tuple:
        return None
    args = get_args(declared)
    if len(args) != 2 or Ellipsis in args:
        return None
    return args[0], args[1]


def type_text(type_: object) -> str:
    """One operand's type, short enough to sit on a report line."""
    tensors = utils.tensor_types(type_)
    if not tensors:
        return str(type_)
    first = tensors[0]
    shape = ",".join(str(dim) for dim in first.shape)
    more = "+" if len(tensors) > 1 else ""
    storage = f" {first.storage}"
    return f"{first.dtype.name}[{shape}]{more}{storage}"


def operand_rows(record: analysis.MemoryMetadata, node: object, *, printer) -> list[dict[str, object]] | None:
    """Each recorded amount, against the operand it was charged to."""
    if not isinstance(node, expr.Call) or not record.operands:
        return None
    operands = (*node.args, node)
    return [
        {
            "arg": "result" if index == len(node.args) else index,
            "name": printer.print(operand),
            "type": type_text(operand.type),
            "read": moved.read,
            "write": moved.write,
        }
        for index, (operand, moved) in enumerate(zip(operands, record.operands))
    ]


def records_of(node: object, selected: frozenset[type[expr.IRMetadata]], printer) -> dict[str, object]:
    """Every selected record on *expr*, keyed by the family that owns it."""
    result: dict[str, object] = {}
    for metadata_type in sorted(selected, key=lambda record_type: record_type.__name__):
        record = expr.get_metadata(node, metadata_type)
        if record is not None and metadata_type.family is not None:
            result[metadata_type.family] = render_record(record, node, printer=printer)
    return result


def selected_types(
    module: core.Module,
    analyses: tuple[str, ...],
    metadata_types: tuple[type[expr.IRMetadata], ...],
) -> tuple[type[expr.IRMetadata], ...]:
    """Return record types requested analyses own and actually wrote."""
    target = module.resolve_target()
    owned: set[type[expr.IRMetadata]] = set()
    for name in analyses:
        owned.update(target.get_analyzer(name).produces)
    return tuple(dict.fromkeys(item for item in metadata_types if item in owned))


def report_data(
    *,
    module: core.Module,
    function: core.Function,
    analyses: tuple[str, ...],
    topology_level: str | None,
    executed: tuple[str, ...],
    metadata_types: tuple[type[expr.IRMetadata], ...],
    labels: Mapping[int, str] | None = None,
    printer=None,
    functions: tuple[core.Function, ...] = (),
) -> dict[str, object]:
    """Project every definition once, in root-first order."""
    if printer is None:
        printer = HirPrinter()
        for ctx in reversed(valuation.definitions(valuation.Context(module, function))):
            printer.render(ctx.function)
    selected = frozenset(selected_types(module, analyses, metadata_types))
    target = module.resolve_target()
    owners = {
        id(candidate): frame.module.name
        for frame in valuation.list_call_paths(module).values()
        for candidate in (frame.function, *getattr(frame.function, "variants", ()))
    }
    rows = []
    for body in functions or (function,):
        local_printer = printer.definition_printer(body)
        row = {
            "function": f"{owners.get(id(valuation.origin_of(body) or body), module.name)}.{body.name}",
            "function_records": records_of(body, selected, local_printer),
            "calls": [],
            "loops": [],
            "meshes": [],
        }
        for node in core.collect_exprs(body.body):
            classifier = ROWS.get(type(node))
            if classifier is None:
                continue
            held = records_of(node, selected, local_printer)
            if not held:
                continue
            kind, label = classifier
            value = (labels.get(id(node)) if labels is not None else None) or (local_printer.print(node) if isinstance(node, expr.Call) else label(node, row))
            row[kind].append({"value": value, **held})
        asked = {analysis.ComputeCostMetadata, analysis.MemoryMetadata, analysis.RegionMemoryMetadata}
        if asked & selected or ("roofline" in analyses and asked & set(metadata_types)):
            row["totals"] = work_totals(body)
        rows.append(row)
    return {
        "target": target.identity,
        "module": module.name,
        "topology": topology_level,
        "requested": list(analyses),
        "executed": list(executed),
        "functions": rows,
    }


def totals_of(held: "analysis.Breakdown | analysis.Spread | None") -> dict[str, object]:
    """Each kind's whole-program amount, without any level's share."""
    if held is None:
        return {}
    if isinstance(held, analysis.Spread):
        return dict(held.total)
    return {
        kind: reported_value(spread.total, type(spread.total))
        if is_dataclass(type(spread.total))
        else spread.total
        for kind, spread in held.kinds
    }


def work_totals(function: core.Function) -> dict[str, object]:
    """The multiplicity-aware work and movement recorded on the Function root.

    Two families answer here, each about its own half, so a report states what
    a program asks of the machine beside what it moves through it.
    """
    record = expr.get_metadata(function, analysis.ComputeCostMetadata)
    moved = expr.get_metadata(function, analysis.RegionMemoryMetadata)
    return {
        "flops": totals_of(None if record is None else record.flops),
        "traffic": totals_of(None if moved is None else moved.traffic.storage),
        "communication": totals_of(None if moved is None else moved.traffic.communication),
    }


def render_json(data: dict[str, object]) -> str:
    """Render report data as sorted JSON for byte-stable comparisons."""
    return json.dumps(data, indent=2, sort_keys=True)


EXPR_FIELDS: Mapping[type[expr.IRMetadata], Mapping[str, Callable[..., object]]] = MappingProxyType(
    {analysis.MemoryMetadata: MappingProxyType({"operands": operand_rows})}
)
"""The report fields of each record type that are read from its expression."""

ROWS = MappingProxyType(
    {
        expr.Call: ("calls", None),
        core.LoopRegion: ("loops", lambda node, row: node.induction_var.name),
        core.MeshRegion: ("meshes", lambda node, row: f"mesh{len(row['meshes'])}"),
    }
)
"""Where each node kind sits in a function's report row, and how an unlabelled one is named."""


__all__ = [
    "render_json",
    "render_record",
    "report_data",
    "selected_types",
    "type_text",
]
