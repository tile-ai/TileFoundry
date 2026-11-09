"""Registered HIR-to-TIR candidate pairings."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping

from tilefoundry import dispatch
from tilefoundry.ir import expr, pattern

from .instruction import RefusalError


@dataclass(frozen=True)
class HirMatch:
    """A TIR declaration's HIR semantics, fixed modes and operand roles."""

    op: type
    when: Mapping[str, object] = field(default_factory=dict)
    attributes: Mapping[str, object] = field(default_factory=dict)
    operands: Mapping[str, str] = field(default_factory=dict)


def hir_declarations(hir_op):
    for schema in expr.iter_schemas():
        if schema.dialect != "T" or schema.op_class is None:
            continue
        for declaration in schema.op_class.__dict__.get("hir_matches", ()):
            if declaration.op is hir_op:
                yield schema.op_class, declaration


class Pairing:
    """The TIR instructions registered for one HIR operation, and how to admit them."""

    def __init__(self, hir_op: type) -> None:
        self.hir_op = hir_op
        self.ops: list[type] = []
        self.when: dict[type, Callable] = {}
        self.callback: Callable | None = None

    def add(self, tir_ops: tuple[type, ...], when: Callable | None) -> None:
        """Append *tir_ops* once each; *when* replaces the admission test of each."""
        for tir_op in tir_ops:
            if tir_op not in self.ops:
                self.ops.append(tir_op)
            if when is not None:
                self.when[tir_op] = when

    def set_callback(self, callback: Callable) -> None:
        """Keep the one callback that supplies attributes and operand types."""
        if self.callback is not None:
            raise ValueError(f"{self.hir_op.__name__} already has a candidate operand callback")
        self.callback = callback

    def admits(self, tir_op: type, call) -> bool:
        """Whether *tir_op* is a candidate for *call*, which only *when* can refuse."""
        when = self.when.get(tir_op)
        return when is None or bool(when(call))


pairings: dispatch.Registry[type, Pairing] = dispatch.Registry("candidate_pairings")


def register_candidates(hir_op: type, tir_ops: tuple[type, ...], *, when: Callable | None = None):
    """Append instructions; a decorated callback supplies attributes and operand types.

    *when* says which calls the instructions carry at all; on any other call
    they are no candidate.
    """
    pairing = pairings.get(hir_op, None)
    if pairing is None:
        pairing = Pairing(hir_op)
        pairings.register(hir_op, pairing)
    pairing.add(tir_ops, when)

    def decorate(callback):
        pairing.set_callback(callback)
        return callback

    return decorate


def candidate_ops(hir_op: type, *, target=None, call=None) -> tuple[type, ...]:
    """Return every registered TIR carrier for one HIR operation, or for one *call* of it."""
    supported = None if target is None else {op for op, _ in expr.supported_op_capabilities(target)}
    pairing = pairings.get(hir_op, None)
    registered = () if pairing is None else pairing.ops
    return tuple(
        op
        for op in dict.fromkeys((*registered, *(op for op, _ in hir_declarations(hir_op))))
        if (supported is None or op in supported)
        and (call is None or pairing is None or pairing.admits(op, call))
    )


def instruction_attributes(hir_op: expr.Op, op_type: type[expr.Op]) -> dict:
    """Copy same-named attributes and reject information the instruction cannot express."""
    parameters = expr.collect_param_defs(op_type)
    names = {param.name for param in parameters}
    for param in expr.collect_param_defs(type(hir_op)):
        if param.kind != "attribute" or param.name in names:
            continue
        value = getattr(hir_op, param.name)
        if not param.has_default or value != param.default:
            raise RefusalError(
                f"{type(hir_op).__name__}.{param.name}={value!r} has no counterpart on {op_type.__name__}"
            )
    return {
        param.name: getattr(hir_op, param.name)
        for param in parameters
        if param.kind == "attribute" and hasattr(hir_op, param.name)
    }


def instruction_from_hir(hir_op: expr.Op, op_type: type[expr.Op]) -> expr.Op | None:
    """Build an instruction from same-named HIR attributes, or None if incomplete."""
    attributes = instruction_attributes(hir_op, op_type)
    if any(
        param.kind == "attribute" and not param.has_default and param.name not in attributes
        for param in expr.collect_param_defs(op_type)
    ):
        return None
    return op_type(**attributes)


def candidate_bindings(call, op_type):
    """Yield applicable data declarations, keeping historical callbacks as fallback."""
    declarations = tuple(d for op, d in hir_declarations(type(call.target)) if op is op_type)
    if not declarations:
        yield legacy_operands(call, op_type)
        return
    parameters = expr.collect_param_defs(op_type)
    source = {
        param.name: arg.type
        for param, arg in zip(
            (p for p in expr.collect_param_defs(type(call.target)) if p.kind == "input"),
            call.args, strict=True,
        )
    }
    source["result"] = call.type
    for declaration in declarations:
        condition = pattern.Matcher()
        if not all(condition.match(value, getattr(call.target, name)) for name, value in declaration.when.items()):
            continue
        if condition.solve() is not True:
            continue
        attributes = dict(declaration.attributes)
        if any(name in condition.bindings and condition.bindings[name] != value
               for name, value in attributes.items()):
            continue
        if any(
            p.effect & expr.MemoryEffect.WRITE and declaration.operands.get(p.name) not in (None, "result")
            and source[declaration.operands[p.name]] != call.type
            for p in parameters if p.kind == "input"
        ):
            continue
        matcher = pattern.Matcher({**condition.bindings, **attributes})
        types = tuple(source[declaration.operands[p.name]] for p in parameters if p.kind == "input" and p.name in declaration.operands)
        inputs = tuple(p for p in parameters if p.kind == "input" and p.name in declaration.operands)
        for param, type_ in zip(inputs, types, strict=True):
            def fields(declared):
                if isinstance(declared, pattern.Tensor):
                    return replace(declared, shape=None, layout=None)
                if isinstance(declared, pattern.Switch):
                    branches = {key: fields(value) for key, value in declared.branches if value is not None}
                    values = tuple(branches.values())
                    if values and all(pattern.same_structure(value, values[0]) for value in values):
                        return values[0]
                    return pattern.Switch(declared.param, branches)
                return declared
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
        raise RefusalError(f"{op_type.__name__} has no applicable HIR declaration") from None


def legacy_operands(call, op_type):
    """The registered callback, or the default instruction-order operand mapping."""
    pairing = pairings.get(type(call.target), None)
    callback = None if pairing is None else pairing.callback
    if callback is not None:
        return callback(call, op_type)
    attributes = instruction_attributes(call.target, op_type)
    reads = iter(arg.type for arg in call.args)
    operands = tuple(
        call.type if param.effect & expr.MemoryEffect.WRITE else next(reads)
        for param in expr.collect_param_defs(op_type)
        if param.kind == "input" and not param.optional
    )
    return attributes, operands


def sole_candidate(hir_op: expr.Op, *, target=None) -> expr.Op | None:
    """Build the one registered instruction from same-named attributes, or None."""
    call = hir_op if isinstance(hir_op, expr.Call) else None
    hir_target = call.target if call is not None else hir_op
    candidates = candidate_ops(type(hir_target), target=target, call=call)
    if len(candidates) != 1:
        return None
    if call is None:
        return instruction_from_hir(hir_target, candidates[0])
    attributes, _ = candidate_operands(call, candidates[0])
    if any(
        param.kind == "attribute" and not param.has_default and param.name not in attributes
        for param in expr.collect_param_defs(candidates[0])
    ):
        return None
    return candidates[0](**attributes)


__all__ = [
    "HirMatch",
    "Pairing",
    "candidate_bindings",
    "candidate_operands",
    "candidate_ops",
    "instruction_from_hir",
    "pairings",
    "register_candidates",
    "sole_candidate",
]
