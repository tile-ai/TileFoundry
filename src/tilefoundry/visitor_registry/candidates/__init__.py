"""Registered HIR-to-TIR candidate pairings."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field, replace
from typing import Mapping

from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.core.op import Op, supported_op_capabilities
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.ir.core.param_def import MemoryEffect, collect_param_defs
from tilefoundry.ir.hir.schedule import ScheduleRefusal
from tilefoundry.ir.pattern import PatternMatcher, SwitchPattern, TensorPattern
from tilefoundry.ir.pattern.predicates import _same


@dataclass(frozen=True)
class HirMatch:
    """A TIR declaration's HIR semantics, fixed modes and operand roles."""

    op: type
    when: Mapping[str, object] = field(default_factory=dict)
    attributes: Mapping[str, object] = field(default_factory=dict)
    operands: Mapping[str, str] = field(default_factory=dict)


def _declarations(hir_op):
    for schema in iter_schemas():
        if schema.dialect != "T" or schema.op_class is None:
            continue
        for declaration in schema.op_class.__dict__.get("hir_matches", ()):
            if declaration.op is hir_op:
                yield schema.op_class, declaration


_CANDIDATES: dict[type, list[type]] = {}
_CALLBACKS: dict[type, object] = {}


def register_candidates(hir_op: type, tir_ops: tuple[type, ...]):
    """Append instructions; a decorated callback supplies attributes and operand types."""
    registered = _CANDIDATES.setdefault(hir_op, [])
    for tir_op in tir_ops:
        if tir_op not in registered:
            registered.append(tir_op)

    def decorate(callback):
        if hir_op in _CALLBACKS:
            raise ValueError(f"{hir_op.__name__} already has a candidate operand callback")
        _CALLBACKS[hir_op] = callback
        return callback

    return decorate


def candidate_ops(hir_op: type, *, target=None) -> tuple[type, ...]:
    """Return every registered TIR carrier for one HIR operation."""
    supported = None if target is None else {op for op, _ in supported_op_capabilities(target)}
    return tuple(op for op in dict.fromkeys(
        (*_CANDIDATES.get(hir_op, ()), *(op for op, _ in _declarations(hir_op)))
    ) if supported is None or op in supported)


def _instruction_attributes(hir_op: Op, op_type: type[Op]) -> dict:
    """Copy same-named attributes and reject information the instruction cannot express."""
    parameters = collect_param_defs(op_type)
    names = {param.name for param in parameters}
    for param in collect_param_defs(type(hir_op)):
        if param.kind != "attribute" or param.name in names:
            continue
        value = getattr(hir_op, param.name)
        if not param.has_default or value != param.default:
            raise ScheduleRefusal(
                f"{type(hir_op).__name__}.{param.name}={value!r} has no counterpart on {op_type.__name__}"
            )
    return {
        param.name: getattr(hir_op, param.name)
        for param in parameters
        if param.kind == "attribute" and hasattr(hir_op, param.name)
    }


def instruction_from_hir(hir_op: Op, op_type: type[Op]) -> Op | None:
    """Build an instruction from same-named HIR attributes, or None if incomplete."""
    attributes = _instruction_attributes(hir_op, op_type)
    if any(
        param.kind == "attribute" and not param.has_default and param.name not in attributes
        for param in collect_param_defs(op_type)
    ):
        return None
    return op_type(**attributes)


def candidate_bindings(call, op_type):
    """Yield applicable data declarations, keeping historical callbacks as fallback."""
    declarations = tuple(d for op, d in _declarations(type(call.target)) if op is op_type)
    if not declarations:
        yield _legacy_operands(call, op_type)
        return
    parameters = collect_param_defs(op_type)
    source = {
        param.name: arg.type
        for param, arg in zip(
            (p for p in collect_param_defs(type(call.target)) if p.kind == "input"),
            call.args, strict=True,
        )
    }
    source["result"] = call.type
    for declaration in declarations:
        condition = PatternMatcher()
        if not all(condition.match(value, getattr(call.target, name)) for name, value in declaration.when.items()):
            continue
        if condition.solve() is not True:
            continue
        attributes = dict(declaration.attributes)
        if any(name in condition.bindings and condition.bindings[name] != value
               for name, value in attributes.items()):
            continue
        if any(
            p.effect & MemoryEffect.WRITE and declaration.operands.get(p.name) not in (None, "result")
            and source[declaration.operands[p.name]] != call.type
            for p in parameters if p.kind == "input"
        ):
            continue
        matcher = PatternMatcher({**condition.bindings, **attributes})
        types = tuple(source[declaration.operands[p.name]] for p in parameters if p.kind == "input" and p.name in declaration.operands)
        inputs = tuple(p for p in parameters if p.kind == "input" and p.name in declaration.operands)
        for param, type_ in zip(inputs, types, strict=True):
            def fields(pattern):
                if isinstance(pattern, TensorPattern):
                    return replace(pattern, shape=None, layout=None)
                if isinstance(pattern, SwitchPattern):
                    branches = {key: fields(value) for key, value in pattern.branches if value is not None}
                    values = tuple(branches.values())
                    if values and all(_same(value, values[0]) for value in values):
                        return values[0]
                    return SwitchPattern(pattern.param, branches)
                return pattern
            if not matcher.match(fields(param.pattern), type_):
                break
        else:
            attributes.update((p.name, matcher.bindings[p.name]) for p in parameters if p.kind == "attribute" and p.name in matcher.bindings)
            yield attributes, types


def candidate_operands(call, op_type):
    """Return the first applicable declaration for legacy single-instance callers."""
    try:
        return next(candidate_bindings(call, op_type))
    except StopIteration:
        raise ScheduleRefusal(f"{op_type.__name__} has no applicable HIR declaration") from None


def _legacy_operands(call, op_type):
    """The registered callback, or the default instruction-order operand mapping."""
    callback = _CALLBACKS.get(type(call.target))
    if callback is not None:
        return callback(call, op_type)
    attributes = _instruction_attributes(call.target, op_type)
    reads = iter(arg.type for arg in call.args)
    operands = tuple(
        call.type if param.effect & MemoryEffect.WRITE else next(reads)
        for param in collect_param_defs(op_type)
        if param.kind == "input" and not param.optional
    )
    return attributes, operands


def sole_candidate(hir_op: Op, *, target=None) -> Op | None:
    """Build the one registered instruction from same-named attributes, or None."""
    call = hir_op if isinstance(hir_op, Call) else None
    hir_target = call.target if call is not None else hir_op
    candidates = candidate_ops(type(hir_target), target=target)
    if len(candidates) != 1:
        return None
    if call is None:
        return instruction_from_hir(hir_target, candidates[0])
    attributes, _types = candidate_operands(call, candidates[0])
    if any(
        param.kind == "attribute" and not param.has_default and param.name not in attributes
        for param in collect_param_defs(candidates[0])
    ):
        return None
    return candidates[0](**attributes)


def _auto_import(pkg_name: str) -> None:
    package = importlib.import_module(pkg_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, f"{pkg_name}."):
        importlib.import_module(module_name)


_auto_import(__name__)

__all__ = [
    "HirMatch",
    "candidate_bindings",
    "candidate_operands",
    "instruction_from_hir",
    "sole_candidate",
    "candidate_ops",
    "register_candidates",
]
