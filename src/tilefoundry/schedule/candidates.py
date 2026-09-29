"""Instruction candidates for unscheduled HIR matmul and reshard sites."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import groupby
from math import prod
from typing import Any

import isl

from tilefoundry.analysis import analyze
from tilefoundry.analysis.iteration_scope import build_scopes, walk_scopes
from tilefoundry.inspection import PatternPrinter
from tilefoundry.ir.core import (
    Call,
    OpCapability,
    Var,
    get_metadata,
    op_identifier,
    supported_op_capabilities,
    value_label,
)
from tilefoundry.ir.core.metadata import SourceSpanMetadata
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.pattern import (
    PatternMatcher,
    SwitchPattern,
    TensorPattern,
    between_rules,
    declared_execution_mesh,
)
from tilefoundry.ir.types import TensorType, UnitType
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.ir.types.utils import local_type_of
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import Target
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    relation_of,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import FunctionScope, TypeInferContext


@dataclass(frozen=True)
class _Site:
    call: Call
    label: str
    op: str
    scope: str
    reads: tuple[tuple[str, TensorType], ...]
    leaves: tuple[tuple[str, TensorType], ...]


def _instructions(target: Target) -> tuple[tuple[type, OpCapability], ...]:
    """Return discoverable declarations that state comparable coordinates."""
    families = [
        (op_type, tuple(capability for _op_type, capability in entries))
        for op_type, entries in groupby(supported_op_capabilities(target), key=lambda item: item[0])
        if access_relation_registry.lookup(op_type) is not None
    ]
    families.sort(key=lambda item: min(cap.report_order for cap in item[1]))
    return tuple(
        (op_type, capability) for op_type, capabilities in families for capability in capabilities
    )


def _input_params(op_type: type) -> tuple[ParamDef, ...]:
    return tuple(param for param in op_type._op_schema.signature if param.kind == "input")


def _site_types(call: Call) -> tuple[tuple[TensorType, ...], TensorType]:
    reads = tuple(local_type_of(arg.type) for arg in call.args)
    output = local_type_of(call.type)
    if not all(isinstance(type_, TensorType) for type_ in (*reads, output)):
        raise ValueError(f"{type(call.target).__name__} candidate site is not tensor-valued")
    if isinstance(call.target, Reshard):
        shape = tuple(
            min(source, destination)
            for source, destination in zip(reads[0].shape, output.shape, strict=True)
        )
        reads = (replace(reads[0], shape=shape, layout=None),)
        output = replace(output, shape=shape, layout=None)
    return reads, output


def _sites(module, function) -> tuple[_Site, ...]:
    root = build_scopes(module, function)
    owners = {identity: scope for scope in walk_scopes(root) for identity in scope.relations}
    sites = []
    for expr in collect_exprs(function.body):
        if not isinstance(expr, Call) or not isinstance(expr.target, (MatMul, Reshard)):
            continue
        reads, output = _site_types(expr)
        schema = type(expr.target)._op_schema
        names = tuple(param.name for param in schema.signature if param.kind == "input")
        scope = owners[id(expr)]
        mesh = scope.enclosing_mesh()
        level = None if mesh is None else mesh.topologies[-1].name
        sites.append(
            _Site(
                call=expr,
                label=value_label(expr) or schema.name,
                op=f"{schema.dialect}.{schema.name}",
                scope="whole program" if level is None else f"per {level}",
                reads=tuple(zip(names, reads, strict=True)),
                leaves=(("result", output),),
            )
        )
    return tuple(sites)


def _is_projection(relation: isl.map, source_axis: int, target_axis: int) -> bool:
    local = isl.local_space.from_space(relation.get_space())
    equal = isl.constraint.alloc_equality(local)
    equal = equal.set_coefficient_si(isl.dim_type.IN, source_axis, 1)
    equal = equal.set_coefficient_si(isl.dim_type.OUT, target_axis, -1)
    projected = isl.map.universe(relation.get_space()).add_constraint(equal)
    return relation.is_subset(projected)


def _relation_shape(boundary) -> tuple[int, tuple[int | None, ...]]:
    relation = relation_of(boundary.pattern)
    source_rank = relation.dim(isl.dim_type.IN)
    axes = []
    for target_axis in range(relation.dim(isl.dim_type.OUT)):
        sources = tuple(
            source_axis
            for source_axis in range(source_rank)
            if _is_projection(relation, source_axis, target_axis)
        )
        axes.append(sources[0] if len(sources) == 1 else None)
    return source_rank, tuple(axes)


def _site_relation_shape(site: _Site, ctx: TypeInferContext) -> tuple:
    relations = relations_of(site.call, ctx)
    return (
        tuple(_relation_shape(boundary) for boundary in relations.inputs),
        tuple(_relation_shape(boundary) for boundary in relations.outputs),
    )


def _instruction_operands(site: _Site, op) -> tuple[TensorType, ...] | None:
    params = _input_params(type(op))
    read_params = tuple(
        param
        for param in params
        if param.effect & MemoryEffect.READ and not param.effect & MemoryEffect.WRITE
    )
    write_params = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
    if len(read_params) != len(site.reads) or len(write_params) > len(site.leaves):
        return None
    read_types = iter(type_ for _name, type_ in site.reads)
    write_types = iter(type_ for _name, type_ in site.leaves)
    operands = []
    for param in params:
        if param.effect & MemoryEffect.WRITE:
            operands.append(next(write_types))
        elif param.effect & MemoryEffect.READ:
            operands.append(next(read_types))
    return tuple(operands)


def _instruction_relation_shape(site: _Site, op) -> tuple | None:
    types = _instruction_operands(site, op)
    if types is None:
        return None
    args = tuple(Var(name=f"operand{index}", type=type_) for index, type_ in enumerate(types))
    call = Call(target=op, args=args, type=UnitType())
    relations = relations_of(call, TypeInferContext())
    params = _input_params(type(op))
    reads = tuple(
        _relation_shape(boundary)
        for param, boundary in zip(params, relations.inputs, strict=True)
        if param.effect & MemoryEffect.READ and not param.effect & MemoryEffect.WRITE
    )
    writes = tuple(
        _relation_shape(boundary)
        for param, boundary in zip(params, relations.inputs, strict=True)
        if param.effect & MemoryEffect.WRITE
    )
    return reads, writes


def _parameter_values(param: ParamDef, site: _Site) -> tuple:
    if param.has_default:
        return ()
    annotation = param.annotation
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(annotation)
    if annotation is int:
        largest = max(
            extent
            for _name, type_ in (*site.reads, *site.leaves)
            for extent in type_.shape
            if isinstance(extent, int) and not isinstance(extent, bool)
        )
        return tuple(range(1, largest + 1))
    return ()


def _variant_instances(
    op_type: type,
    capability: OpCapability,
    site: _Site,
) -> tuple[tuple[object | None, dict], ...]:
    if capability.attribute is None:
        return ((None, {}),)
    declaration = capability.declaration
    states: tuple[dict, ...] = ({},)
    for param in declaration.parameters:
        if param.has_default:
            continue
        held = []
        for state in states:
            for value in _parameter_values(param, site):
                matcher = PatternMatcher(state)
                if matcher.match(param.pattern, value) and matcher.solve():
                    held.append({**state, param.name: value})
        states = tuple(held)
    variants = []
    for state in states:
        try:
            variants.append((declaration(**state), state))
        except ValueError:
            continue
    return tuple(variants)


def _instantiate(op_type: type, capability: OpCapability, variant):
    if capability.attribute is None:
        return op_type()
    return op_type(**{capability.attribute: variant})


def _selected(pattern, bindings: dict):
    while isinstance(pattern, SwitchPattern):
        if pattern.param not in bindings:
            return None
        pattern = dict(pattern.branches).get(bindings[pattern.param])
    return pattern


def _operand_pattern(param: ParamDef, op, bindings: dict) -> TensorPattern | None:
    pattern = param.pattern
    if hasattr(pattern, "read_on"):
        pattern = pattern.read_on(op)
    pattern = _selected(pattern, bindings)
    return pattern if isinstance(pattern, TensorPattern) else None


def _declared_shape(pattern: TensorPattern, bindings: dict) -> tuple[int, ...] | None:
    if pattern.shape is None:
        return None
    shape = []
    for dim in pattern.shape:
        if isinstance(dim, int) and not isinstance(dim, bool):
            shape.append(dim)
            continue
        name = getattr(dim, "name", None)
        value = bindings.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        shape.append(value)
    return tuple(shape)


def _field_name(value) -> str:
    return getattr(value, "name", str(value)).lower()


def _field_refusals(
    name: str,
    pattern: TensorPattern,
    type_: TensorType,
    bindings: dict,
) -> list[str]:
    printer = PatternPrinter()
    refused = []
    for field in ("dtype", "storage"):
        wanted = getattr(pattern, field)
        actual = getattr(type_, field)
        matcher = PatternMatcher(bindings)
        if matcher.match(wanted, actual) and matcher.solve():
            continue
        if isinstance(wanted, Enum) or hasattr(wanted, "name") and not hasattr(wanted, "match"):
            written = _field_name(wanted)
        else:
            written = printer.written(wanted, field)
        refused.append(f"{name} {field}={_field_name(actual)}, reads {field}={written}")
    return refused


def _type_refusals(
    name: str,
    pattern: TensorPattern | None,
    whole: TensorType,
    bindings: dict,
) -> tuple[TensorType, tuple[str, ...]]:
    if pattern is None:
        return whole, (f"{name} states no tensor pattern",)
    shape = _declared_shape(pattern, bindings)
    if shape is None:
        shape = tuple(whole.shape)
    single = TensorType(shape, whole.dtype, None, whole.storage)
    simplified = replace(pattern, layout=None)
    matcher = PatternMatcher(bindings)
    if matcher.match(simplified, single) and matcher.solve():
        return single, ()
    refused = _field_refusals(name, simplified, single, bindings)
    if not refused:
        refused.append(f"{name}: {PatternPrinter().refusal(matcher.refusal)}")
    return single, tuple(refused)


def _whole_tiles(site: _Site, op, variant) -> bool:
    bindings = dict(getattr(variant, "bindings", {}))
    asked = _asked(site, op)
    if asked is None:
        return False
    for param, whole in asked:
        pattern = _operand_pattern(param, op, bindings)
        if pattern is None:
            return False
        shape = _declared_shape(pattern, bindings)
        if shape is None:
            continue
        if len(shape) != len(whole.shape) or any(
            not isinstance(extent, int) or extent % atom_extent
            for extent, atom_extent in zip(whole.shape, shape, strict=True)
        ):
            return False
    return True


def _pattern_refusals(site: _Site, op, variant) -> tuple[str, ...]:
    bindings = dict(getattr(variant, "bindings", {}))
    asked = _asked(site, op)
    if asked is None:
        return ("operand counts differ",)
    given = {}
    refused = []
    for param, whole in asked:
        pattern = _operand_pattern(param, op, bindings)
        if pattern is None:
            refused.append(f"{param.name} states no tensor pattern")
            continue
        simplified = replace(pattern, layout=None)
        refused.extend(_field_refusals(param.name, simplified, whole, bindings))
        given[param.name] = whole
    for rule in between_rules(type(op)):
        if rule.field in ("storage", "dtype") and not rule.holds(given):
            refused.append(rule.refused(given))
    return tuple(refused)


def _asked(site: _Site, op) -> tuple[tuple[ParamDef, TensorType], ...] | None:
    params = _input_params(type(op))
    read_params = tuple(param for param in params if param.effect == MemoryEffect.READ)
    write_params = tuple(param for param in params if param.effect == MemoryEffect.WRITE)
    if len(read_params) != len(site.reads) or len(write_params) > len(site.leaves):
        return None
    return (
        *zip(read_params, (type_ for _name, type_ in site.reads), strict=True),
        *zip(
            write_params,
            (type_ for _name, type_ in site.leaves[: len(write_params)]),
            strict=True,
        ),
    )


def _refusals(site: _Site, op, variant) -> tuple[str, ...]:
    bindings = dict(getattr(variant, "bindings", {}))
    asked = _asked(site, op)
    if asked is None:
        return ("operand counts differ",)
    given = {}
    refused = []
    for param, whole in asked:
        pattern = _operand_pattern(param, op, bindings)
        single, reasons = _type_refusals(param.name, pattern, whole, bindings)
        given[param.name] = single
        refused.extend(reasons)
    for rule in between_rules(type(op)):
        if rule.field not in ("storage", "dtype"):
            continue
        if not rule.holds(given):
            refused.append(rule.refused(given))
    return tuple(refused)


def _fixed_execution_mesh_size(pattern) -> int | None:
    layout = getattr(pattern, "layout", None)
    outer = getattr(layout, "outer", layout)
    shape = tuple(flatten(getattr(outer, "shape", ())))
    return prod(shape) if shape and all(type(extent) is int for extent in shape) else None


def _needs(site: _Site, op, variant) -> str | None:
    pattern = declared_execution_mesh(type(variant) if variant is not None else type(op))
    size = _fixed_execution_mesh_size(pattern)
    if size is None:
        return None
    groups = 1
    if variant is not None:
        asked = _asked(site, op)
        first_read = None if asked is None else next(iter(asked), None)
        if first_read is not None:
            param, whole = first_read
            bindings = dict(getattr(variant, "bindings", {}))
            operand_pattern = _operand_pattern(param, op, bindings)
            declared = (
                None if operand_pattern is None else _declared_shape(operand_pattern, bindings)
            )
            if declared is not None and whole.shape[0] % declared[0] == 0:
                groups = whole.shape[0] // declared[0]
    topology = pattern.topologies[0]
    return f"{topology} p0:p0+{size * groups}, p0 % {size} = 0"


def _written_bindings(bindings: dict) -> str:
    return ", ".join(f"{name}={getattr(value, 'name', value)}" for name, value in bindings.items())


def _common(groups: list[tuple[str, ...]]) -> list[str]:
    sets = [set(group) for group in groups]
    if not sets:
        return []
    shared = set.intersection(*sets)
    return sorted(shared if shared else set().union(*sets))


def _written_operand(name: str, type_: TensorType) -> str:
    return f"{name}={tuple(type_.shape)} {type_.dtype.name} {type_.storage}"


def _source_label(sites: tuple[_Site, ...], module, source: str | None) -> str:
    if source is not None:
        return source
    for site in sites:
        span = get_metadata(site.call, SourceSpanMetadata)
        if span is not None:
            return span.file
    return module.name


def candidates(module, entry, *, source: str | None = None) -> dict[str, Any]:
    """Report instruction candidates for every unscheduled supported HIR site."""
    result = analyze(module, entry, analysis=("memory",))
    sites = _sites(result.module, result.function)
    if not sites:
        raise ValueError("source has no unscheduled matmul or reshard candidate site")
    target = result.module.resolve_target()
    declared = _instructions(target)
    ctx = TypeInferContext(scope=FunctionScope(result.module, result.function))
    rows = []
    for site in sites:
        site_shape = _site_relation_shape(site, ctx)
        usable, refused = [], []
        handed_result = False
        for op_type, capability in declared:
            variants = _variant_instances(op_type, capability, site)
            if not variants:
                continue
            prototype, _binding = variants[0]
            op = _instantiate(op_type, capability, prototype)
            if _instruction_relation_shape(site, op) != site_shape:
                continue
            if any(param.effect == MemoryEffect.WRITE for param in _input_params(type(op))):
                handed_result = True
            accepted, reasons, needs = [], [], []
            for variant, binding in variants:
                held = _instantiate(op_type, capability, variant)
                if not _whole_tiles(site, held, variant):
                    reasons.append(_pattern_refusals(site, held, variant))
                    continue
                rejected = _refusals(site, held, variant)
                if rejected:
                    reasons.append(rejected)
                    continue
                accepted.append(_written_bindings(binding))
                needs.append(_needs(site, held, variant))
            if accepted:
                usable.append(
                    {
                        "id": op_identifier(capability.declaration or op_type),
                        "needs": "; ".join(dict.fromkeys(filter(None, needs))) or None,
                        "bindings": [binding for binding in accepted if binding],
                    }
                )
            else:
                common = _common(reasons)
                if common:
                    refused.append(
                        {
                            "id": op_identifier(capability.declaration or op_type),
                            "refused": common,
                        }
                    )
        operands = [
            *(_written_operand(name, type_) for name, type_ in site.reads),
            *(_written_operand(name, type_) for name, type_ in site.leaves if handed_result),
        ]
        rows.append(
            {
                "line": site.label,
                "op": site.op,
                "scope": site.scope,
                "operands": operands,
                "candidates": usable,
                "refused": refused,
            }
        )
    return {
        "source": _source_label(sites, result.module, source),
        "target": target.identity,
        "lines": rows,
    }


def render(data: dict[str, Any]) -> str:
    """Render one candidate report as stable text."""
    lines = [f"source {data['source']}", f"target {data['target']}"]
    for row in data["lines"]:
        lines.append(f"  {row['line']}  {row['op']}  {row['scope']}  {'  '.join(row['operands'])}")
        for fit in row["candidates"]:
            needs = "" if fit["needs"] is None else f"  needs {fit['needs']}"
            lines.append(f"    candidate   {fit['id']}{needs}")
            lines.extend(f"                  {binding}" for binding in fit["bindings"])
        for rejection in row["refused"]:
            lines.extend(
                f"    refused     {rejection['id']}: {reason}" for reason in rejection["refused"]
            )
    return "\n".join(lines)


__all__ = ["candidates", "render"]
