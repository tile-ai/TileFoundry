"""Renderings of what an analysis found.

An analysis leaves a semantic result and typed records on the IR. What a human
or a tool reads is a *rendering* of those, produced here rather than by the
analyses: a family that formatted its own text would be deciding presentation,
and two families would then disagree about it.

Text and JSON are two renderings of one report, built once. That is what makes
them carry the same conclusions rather than two independently maintained views
that drift.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TrafficBytes,
)
from tilefoundry.analysis.api import AnalysisResult
from tilefoundry.analysis.registry import ANALYSES
from tilefoundry.analysis.walk import postorder, tensor_types
from tilefoundry.ir.core import Call, IRMetadata, binding_name, get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import bound_dims_of, origin_of


def _traffic(traffic: tuple[tuple[str, TrafficBytes], ...]) -> dict[str, dict[str, int]]:
    return {
        level: {"read": value.read, "write": value.write}
        for level, value in traffic
    }


def _type_text(type_: object) -> str:
    """One operand's type, short enough to sit on a report line."""
    tensors = tensor_types(type_)
    if not tensors:
        return str(type_)
    first = tensors[0]
    shape = ",".join(str(dim) for dim in first.shape)
    more = "+" if len(tensors) > 1 else ""
    storage = "" if first.storage is None else f" {first.storage}"
    return f"{first.dtype.name}[{shape}]{more}{storage}"


def _operand_name(operand: object) -> str:
    """What to call this operand, from the program rather than from the record."""
    name = binding_name(operand) or getattr(operand, "name", None)
    if name:
        return name
    if isinstance(operand, Call):
        return type(operand.target).__name__.lower()
    return type(operand).__name__.lower()


def _operands(record: ComputeCostMetadata, expr: object) -> list[dict[str, object]]:
    """Each recorded amount, against the operand it was charged to."""
    if not isinstance(expr, Call):
        return []
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


def _compute_cost(record: ComputeCostMetadata, expr: object) -> dict[str, object]:
    projected: dict[str, object] = {
        "flops": dict(record.flops),
        "traffic": _traffic(record.traffic),
        "execution_count": record.execution_count,
    }
    operands = _operands(record, expr)
    if operands:
        projected["operands"] = operands
    return projected


def _roofline(record: RooflineMetadata, expr: object) -> dict[str, object]:
    return {
        "compute_ns": record.compute_ns,
        "memory_ns": record.memory_ns,
        "theoretical_ns": record.theoretical_ns,
        "bound_by": record.bound_by,
    }


def _timeline(record: TimelineMetadata, expr: object) -> dict[str, object]:
    return {
        "grid_units": record.grid_units,
        "waves": record.waves,
        "start_ns": record.start_ns,
        "end_ns": record.end_ns,
    }


def _memory(record: MemoryMetadata, expr: object) -> dict[str, object]:
    return {
        "footprint": [
            {
                "level": item.level,
                "peak_bytes": item.peak_bytes,
                "persistent_bytes": item.persistent_bytes,
                "capacity_bytes": item.capacity_bytes,
            }
            for item in record.footprint
        ],
        "traffic": _traffic(record.traffic),
        "lifetimes": [
            {
                "binding": item.binding,
                "level": item.level,
                "bytes": item.bytes,
                "defined_at": item.defined_at,
                "last_used_at": item.last_used_at,
                "persistent": item.persistent,
            }
            for item in record.lifetimes
        ],
        "advisories": list(record.advisories),
    }


# One renderer per record type, so adding a family adds an entry rather than a
# branch in every format.
_RECORDS: tuple[tuple[str, type[IRMetadata], object], ...] = (
    ("compute-cost", ComputeCostMetadata, _compute_cost),
    ("memory", MemoryMetadata, _memory),
    ("roofline", RooflineMetadata, _roofline),
    ("timeline", TimelineMetadata, _timeline),
)


def _records_of(expr: object, selected: frozenset[type[IRMetadata]]) -> dict[str, object]:
    """Every selected record on *expr*, keyed by the family that owns it."""
    result: dict[str, object] = {}
    for name, metadata_type, project in _RECORDS:
        if metadata_type not in selected:
            continue
        record = get_metadata(expr, metadata_type)
        if record is not None:
            result[name] = project(record, expr)
    return result


def _call_label(call: Call, index: int) -> str:
    return binding_name(call) or f"<value {index}>"


def selected_types(
    results: Sequence[AnalysisResult],
) -> tuple[type[IRMetadata], ...]:
    """Which record types these results may be shown through.

    A requested root pulls its dependencies in, so records land on the IR that
    nobody asked to see. What is shown is what the *requested* analyses own,
    intersected with what was actually written: ownership comes from the
    registrations rather than a hand-kept table, and the intersection keeps a
    reader from looking for a record that is not there.

    Every rendering of one run goes through here, so the report and the annotated
    IR cannot make this choice differently.
    """
    if not results:
        return ()
    target = results[0].module.resolve_target()
    owned: set[type[IRMetadata]] = set()
    for item in results:
        owned.update(ANALYSES.resolve(target, item.analysis).produces)
    order: list[type[IRMetadata]] = []
    for item in results:
        for metadata_type in item.metadata_types:
            if metadata_type in owned and metadata_type not in order:
                order.append(metadata_type)
    return tuple(order)


def _same_program(candidate: object, function: object) -> bool:
    """Whether *candidate* and *function* are the same program at the same size.

    The same object, or two rebuilds of one function at one size: the same
    recorded origin and the same recorded extents. A rebuild is settled by those
    two, so they are what is compared.

    The extents are compared because the resulting signature does not imply them.
    A dimension can occur only in a loop bound or a body operation's attribute,
    and then two different sizes rebuild into two different programs whose
    parameters and return type are identical -- so a comparison that read the
    signature would report measurements of two sizes as one.
    """
    if candidate is function:
        return True
    origin = origin_of(candidate)
    if origin is None or origin is not origin_of(function):
        return False
    dims = bound_dims_of(candidate)
    return dims is not None and dims == bound_dims_of(function)


def report(results: Sequence[AnalysisResult]) -> dict[str, object]:
    """One report over every analysis run against one program.

    Only the record types the calls actually wrote are read, so a renderer is
    never sent looking for records that are not there.

    The results have to describe the same program. Usually that is one object,
    but an analysis asked about a size builds the program at that size, so
    several analyses at one size hold several rebuilds and share no object.
    Those are accepted when they were rebuilt from the same function at the same
    recorded extents. Both are stamped only by the specialiser, so neither can be
    claimed by something that was not derived, and the extents are compared
    directly rather than read off the resulting signature -- a dimension occurring
    only inside the body leaves the signature unchanged at every size.

    Structural equality would be the obvious test and is not available: an
    operation carries no equality, so two rebuilds of one program are never
    equal however identical they are.
    """
    if not results:
        raise ValueError("an analysis report needs at least one result")
    first = results[0]
    function = first.function
    if any(not _same_program(item.function, function) for item in results):
        raise ValueError(
            "an analysis report covers one function; these results cover several"
        )
    selected = frozenset(selected_types(results))
    executed: list[str] = []
    for item in results:
        for selector in item.executed:
            if selector not in executed:
                executed.append(selector)
    target = first.module.resolve_target()
    data = {
        "target": getattr(target, "name", type(target).__name__),
        "module": first.module.name,
        "function": function.name,
        "requested": [item.analysis for item in results],
        "executed": executed,
        "function_records": _merged_function_records(results, selected),
        "calls": _merged_call_records(results, selected),
    }
    if ComputeCostMetadata in selected:
        data["totals"] = _work_totals(_costed_function(results))
    return data


def _merged_function_records(
    results: Sequence[AnalysisResult], selected: frozenset
) -> dict[str, object]:
    """Every result's own whole-function records, together.

    An analysis records on the program it ran over, and an analysis asked about a
    size builds that program itself -- so several analyses at one size hold several
    rebuilds and annotate several objects. Reading one of them would report that one
    analysis and silently drop the others, which is worse than refusing: the report
    still names every analysis under `executed`, so the missing conclusions read as
    analyses that had nothing to say rather than as conclusions nobody collected.
    """
    records: dict[str, object] = {}
    for item in results:
        records.update(_records_of(item.function, selected))
    return records


def _merged_call_records(
    results: Sequence[AnalysisResult], selected: frozenset
) -> list[dict[str, object]]:
    """Every result's per-Call records, merged by position in the program.

    Position is the correspondence, and it is sound here for the reason the results
    were accepted at all: they are rebuilds of one program at one set of extents, so
    they walk in the same order. Two rebuilds share no object, so identity is not
    available and position is what is left.
    """
    labels: dict[int, str] = {}
    merged: dict[int, dict[str, object]] = {}
    for item in results:
        for index, expr in enumerate(postorder(item.function.body)):
            if not isinstance(expr, Call):
                continue
            records = _records_of(expr, selected)
            if not records:
                continue
            merged.setdefault(index, {}).update(records)
            labels.setdefault(index, _call_label(expr, index))
    return [{"value": labels[index], **merged[index]} for index in sorted(merged)]


def _costed_function(results: Sequence[AnalysisResult]) -> Function:
    """The rebuild whose Calls carry the work records, for the totals to sum.

    Named rather than assumed to be the first: the totals are the exact sum of those
    records, so summing over a rebuild that does not carry them would report a
    program doing no work.
    """
    for item in results:
        if any(
            get_metadata(expr, ComputeCostMetadata) is not None
            for expr in postorder(item.function.body)
        ):
            return item.function
    return results[0].function


def _work_totals(function: Function) -> dict[str, object]:
    """The function's work, summed from the per-Call records.

    Compute cost has no whole-function record precisely because this sum is all
    there would be in one: adding flops and bytes is exact, so recording the
    result would be storing a second copy of it. Summing is therefore the one
    thing this module computes, and it computes nothing else.
    """
    flops: dict[str, int] = {}
    traffic: dict[str, TrafficBytes] = {}
    for expr in postorder(function.body):
        record = get_metadata(expr, ComputeCostMetadata)
        if record is None:
            continue
        for name, value in record.flops:
            flops[name] = flops.get(name, 0) + value
        for level, value in record.traffic:
            current = traffic.get(level, TrafficBytes())
            traffic[level] = TrafficBytes(
                current.read + value.read,
                current.write + value.write,
            )
    return {
        "flops": dict(sorted(flops.items())),
        "traffic": _traffic(tuple(sorted(traffic.items()))),
    }


def _flop_text(flops: dict[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in sorted(flops.items())) or "0"


def _traffic_text(traffic: dict[str, dict[str, int]]) -> str:
    return (
        ", ".join(
            f"{level}=r{value['read']}/w{value['write']}"
            for level, value in sorted(traffic.items())
        )
        or "0"
    )


def _operand_text(operand: dict[str, object]) -> str:
    return (
        f"{operand['arg']}:{operand['name']}="
        f"r{operand['read']}/w{operand['write']}"
    )


def render_text(data: dict[str, object]) -> str:
    """The report as one stable line per conclusion, each prefixed with ``#``.

    A conclusion appears only when a record states it, or -- for the work totals
    -- when it is the exact sum of records that do.
    """
    lines = [
        f"analysis target={data['target']} module={data['module']} "
        f"function={data['function']}",
        f"analyses={','.join(data['requested'])} executed={','.join(data['executed'])}",
    ]
    totals = data.get("totals")
    if totals is not None:
        lines.append(f"flops {_flop_text(totals['flops'])}")
        lines.append(f"traffic {_traffic_text(totals['traffic'])}")
    records = data["function_records"]
    if "memory" in records:
        memory = records["memory"]
        lines.append(
            "peak-footprint "
            + (
                ", ".join(
                    f"{item['level']}={item['peak_bytes']}"
                    for item in memory["footprint"]
                )
                or "0"
            )
        )
        lines.extend(f"advisory {note}" for note in memory["advisories"])
    if "roofline" in records:
        bound = records["roofline"]
        lines.append(
            f"theoretical-bound={bound['theoretical_ns']}ns by={bound['bound_by']}"
        )
    if "timeline" in records:
        lines.append(f"theoretical-makespan={records['timeline']['end_ns']}ns")
    for call in data["calls"]:
        # Only what the annotated program does not already say on the value's own
        # line. Its flops, its bound and its placement are all there, against the
        # name that produced them; repeating them here made the report a second
        # copy of the program, and the copy was the one whose labels are ambiguous.
        # The per-operand split is not there -- the annotation carries the sum --
        # so this is where it lives, and text and `--json` still agree.
        operands = call.get("compute-cost", {}).get("operands")
        if operands is None:
            continue
        lines.append(
            f"value={call['value']} operands "
            + ", ".join(_operand_text(operand) for operand in operands)
        )
    return "\n".join(f"# {line}" for line in lines)


def render_json(data: dict[str, object]) -> str:
    """The same report as JSON, sorted so two runs compare byte for byte."""
    return json.dumps(data, indent=2, sort_keys=True)


__all__ = ["render_json", "render_text", "report", "selected_types"]
