"""Instruction candidates for unscheduled HIR matmul and reshard sites."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any, Mapping

import isl

from tilefoundry.analysis import analyze
from tilefoundry.analysis.iteration_scope import build_scopes, walk_scopes
from tilefoundry.inspection import PatternPrinter, PythonPrinter
from tilefoundry.ir.core import (
    Call,
    OpCapability,
    get_metadata,
    op_identifier,
    value_label,
)
from tilefoundry.ir.core.metadata import SourceSpanMetadata
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.schedule import operand_relations
from tilefoundry.ir.pattern import (
    PatternMatcher,
    SwitchPattern,
    TensorPattern,
    between_rules,
    declared_execution_mesh,
)
from tilefoundry.ir.pattern.utils import variants
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import is_dim_op_call
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.ir.types.utils import try_local_type_of
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.schedule._reporting import capability_families
from tilefoundry.target import Target
from tilefoundry.visitor_registry.access_relation import (
    access_relation_registry,
    projected_axes,
    relations_of,
)
from tilefoundry.visitor_registry.candidates import (
    candidate_ops,
    instruction_from_hir,
    sole_candidate,
)
from tilefoundry.visitor_registry.contexts import CostContext, FunctionScope


@dataclass(frozen=True)
class _Site:
    call: Call
    label: str
    op: str
    scope: str
    reads: tuple[tuple[str, TensorType], ...]
    leaves: tuple[tuple[str, TensorType], ...]
    instructions: tuple[type, ...]


def _instructions(target: Target) -> tuple[tuple[type, OpCapability], ...]:
    """Return discoverable declarations that state comparable coordinates."""
    return tuple(
        (op_type, capability)
        for op_type, capabilities in capability_families(target)
        if access_relation_registry.lookup(op_type) is not None
        for capability in capabilities
    )


def _input_params(op_type: type, operand_count: int | None = None) -> tuple[ParamDef, ...]:
    params = tuple(param for param in op_type._op_schema.signature if param.kind == "input")
    if operand_count is None:
        return params
    required_reads = sum(
        bool(param.effect & MemoryEffect.READ) for param in params if not param.optional
    )
    optional_reads = tuple(
        param for param in params if param.optional and param.effect & MemoryEffect.READ
    )
    supplied_optional = operand_count - required_reads
    included = {id(param) for param in optional_reads[: max(0, supplied_optional)]}
    return tuple(param for param in params if not param.optional or id(param) in included)


def _site_types(
    call: Call, ctx: CostContext
) -> tuple[tuple[TensorType, ...], TensorType]:
    def candidate_type(type_):
        """Project a site unless it is already one indivisible scheduled issue."""
        projected = try_local_type_of(type_)
        return type_ if projected is None else projected

    reads = tuple(candidate_type(arg.type) for arg in call.args)
    output = candidate_type(call.type)
    if not all(isinstance(type_, TensorType) for type_ in (*reads, output)):
        raise ValueError(f"{type(call.target).__name__} candidate site is not tensor-valued")
    relations = relations_of(call, ctx)
    same_coordinates = (
        len(reads) == 1
        and len(relations) == 2
        and relations[0].relation.is_equal(relations[1].relation)
    )
    if same_coordinates:
        shape = tuple(
            min(source, destination)
            for source, destination in zip(reads[0].shape, output.shape, strict=True)
        )
        reads = (replace(reads[0], shape=shape, layout=None),)
        output = replace(output, shape=shape, layout=None)
    return reads, output


def _sites(module, function, ctx: CostContext) -> tuple[_Site, ...]:
    root = build_scopes(module, function)
    owners = {identity: scope for scope in walk_scopes(root) for identity in scope.relations}
    sites = []
    for expr in collect_exprs(function.body):
        if not isinstance(expr, Call):
            continue
        if is_dim_op_call(expr) or not isinstance(expr.type, TensorType):
            continue
        instructions = candidate_ops(type(expr.target))
        if not instructions:
            continue
        if isinstance(expr.target, HirBinary) and expr.type.shape == ():
            continue
        reads, output = _site_types(expr, ctx)
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
                instructions=instructions,
            )
        )
    return tuple(sites)


def _relation_shape(boundary) -> tuple[int, tuple[int | None, ...]]:
    return boundary.relation.dim(isl.dim_type.IN), projected_axes(boundary)


def _site_relation_shape(site: _Site) -> tuple:
    reads = tuple(type_ for _name, type_ in site.reads)
    relations = operand_relations(site.call.target, reads)
    return (
        tuple(_relation_shape(boundary) for boundary in relations[: len(reads)]),
        tuple(_relation_shape(boundary) for boundary in relations[len(reads) :]),
    )


def _instruction_operands(site: _Site, op) -> tuple[TensorType, ...] | None:
    params = _input_params(type(op), len(site.reads))
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
    relations = operand_relations(op, types)
    params = _input_params(type(op), len(site.reads))
    operands = relations[: len(types)]
    reads = tuple(
        _relation_shape(boundary)
        for param, boundary in zip(params, operands, strict=True)
        if param.effect & MemoryEffect.READ and not param.effect & MemoryEffect.WRITE
    )
    writes = tuple(
        _relation_shape(boundary)
        for param, boundary in zip(params, operands, strict=True)
        if param.effect & MemoryEffect.WRITE
    )
    return reads, writes


def _site_integer_parameter_values(param: ParamDef, site: _Site) -> tuple:
    if param.annotation is not int:
        return ()
    largest = max(
        extent
        for _name, type_ in (*site.reads, *site.leaves)
        for extent in type_.shape
        if isinstance(extent, int) and not isinstance(extent, bool)
    )
    return tuple(range(1, largest + 1))


def _variant_instances(
    op_type: type,
    capability: OpCapability,
    site: _Site,
) -> tuple[tuple[object | None, dict], ...]:
    if capability.attribute is None:
        instruction = instruction_from_hir(site.call.target, op_type)
        return () if instruction is None else ((instruction, {}),)
    declaration = capability.declaration
    states = variants(
        declaration.parameters,
        vary_defaulted=False,
        values=lambda param: _site_integer_parameter_values(param, site),
    )
    instances = []
    for state in states:
        try:
            instances.append((declaration(**state), state))
        except ValueError:
            continue
    return tuple(instances)


def _instantiate(op_type: type, capability: OpCapability, variant):
    if capability.attribute is None:
        return variant
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
    return single, (f"{name}: {PatternPrinter().refusal(matcher.refusal)}",)


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


def _asked(site: _Site, op) -> tuple[tuple[ParamDef, TensorType], ...] | None:
    params = _input_params(type(op), len(site.reads))
    read_params = tuple(param for param in params if param.effect == MemoryEffect.READ)
    write_params = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
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


def _source_label(sites: tuple[_Site, ...], module, source: str | None) -> str:
    if source is not None:
        return source
    for site in sites:
        span = get_metadata(site.call, SourceSpanMetadata)
        if span is not None:
            return span.file
    return module.name


def candidates(
    module,
    entry,
    *,
    source: str | None = None,
    dims: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Report instruction candidates for every unscheduled supported HIR site."""
    result = analyze(module, entry, analysis=("memory",), dims=dims)
    ctx = CostContext(scope=FunctionScope(result.module, result.function))
    sites = _sites(result.module, result.function, ctx)
    if not sites:
        raise ValueError("source has no unscheduled candidate site")
    target = result.module.resolve_target()
    declared = _instructions(target)
    type_printer = PythonPrinter()
    rows = []
    for site in sites:
        site_shape = _site_relation_shape(site)
        automatic = sole_candidate(site.call.target)
        automatic_id = None if automatic is None else op_identifier(type(automatic))
        usable, refused = [], []
        handed_result = False
        for op_type, capability in declared:
            if op_type not in site.instructions:
                continue
            instances = _variant_instances(op_type, capability, site)
            if not instances:
                continue
            prototype, _binding = instances[0]
            op = _instantiate(op_type, capability, prototype)
            if _instruction_relation_shape(site, op) != site_shape:
                continue
            if any(
                param.effect == MemoryEffect.WRITE
                for param in _input_params(type(op), len(site.reads))
            ):
                handed_result = True
            accepted, reasons, needs = [], [], []
            for variant, binding in instances:
                held = _instantiate(op_type, capability, variant)
                rejected = _refusals(site, held, variant)
                if rejected or not _whole_tiles(site, held, variant):
                    reasons.append(rejected)
                    continue
                accepted.append(_written_bindings(binding))
                needs.append(_needs(site, held, variant))
            if accepted:
                candidate = {
                    "id": op_identifier(capability.declaration or op_type),
                    "needs": "; ".join(dict.fromkeys(filter(None, needs))) or None,
                    "bindings": [binding for binding in accepted if binding],
                }
                if candidate["id"] == automatic_id:
                    candidate["default"] = True
                usable.append(candidate)
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
            *(f"{name}={type_printer.print(type_)}" for name, type_ in site.reads),
            *(
                f"{name}={type_printer.print(type_)}"
                for name, type_ in site.leaves
                if handed_result
            ),
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
            kind = "default" if fit.get("default", False) else "candidate"
            needs = "" if fit["needs"] is None else f"  needs {fit['needs']}"
            lines.append(f"    {kind:<12}{fit['id']}{needs}")
            lines.extend(f"                  {binding}" for binding in fit["bindings"])
        for rejection in row["refused"]:
            lines.extend(
                f"    refused     {rejection['id']}: {reason}" for reason in rejection["refused"]
            )
    return "\n".join(lines)


__all__ = ["candidates", "render"]
