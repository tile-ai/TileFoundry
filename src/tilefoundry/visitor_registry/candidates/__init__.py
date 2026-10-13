"""Registered HIR-to-TIR candidate pairings."""

from __future__ import annotations

import importlib
import pkgutil

from tilefoundry.ir.core.op import Op
from tilefoundry.ir.core.param_def import MemoryEffect, collect_param_defs
from tilefoundry.ir.hir.schedule import ScheduleRefusal

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


def candidate_ops(hir_op: type) -> tuple[type, ...]:
    """Return every registered TIR carrier for one HIR operation."""
    return tuple(_CANDIDATES.get(hir_op, ()))


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


def candidate_operands(call, op_type):
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


def sole_candidate(hir_op: Op) -> Op | None:
    """Build the one registered instruction from same-named attributes, or None."""
    candidates = candidate_ops(type(hir_op))
    if len(candidates) != 1:
        return None
    return instruction_from_hir(hir_op, candidates[0])


def _auto_import(pkg_name: str) -> None:
    package = importlib.import_module(pkg_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, f"{pkg_name}."):
        importlib.import_module(module_name)


_auto_import(__name__)

__all__ = [
    "candidate_operands",
    "instruction_from_hir",
    "sole_candidate",
    "candidate_ops",
    "register_candidates",
]
