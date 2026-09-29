"""Registered HIR-to-TIR candidate pairings."""

from __future__ import annotations

import importlib
import pkgutil

from tilefoundry.ir.core.param_def import collect_param_defs

_CANDIDATES: dict[type, list[type]] = {}


def register_candidates(hir_op: type, tir_ops: tuple[type, ...]) -> None:
    """Add the TIR operations that can implement one HIR operation."""
    registered = _CANDIDATES.setdefault(hir_op, [])
    for tir_op in tir_ops:
        if tir_op not in registered:
            registered.append(tir_op)


def candidate_ops(hir_op: type) -> tuple[type, ...]:
    """Return every registered TIR carrier for one HIR operation."""
    return tuple(_CANDIDATES.get(hir_op, ()))


def candidate_attributes(hir_op, tir_op: type) -> dict | None:
    """Map one HIR call's named attributes onto a candidate instruction.

    A candidate remains selectable when every required instruction attribute
    is stated by the HIR operation under the same name. Optional instruction
    attributes keep their own defaults when the HIR operation does not state
    them.
    """
    attributes = {}
    for param in collect_param_defs(tir_op):
        if param.kind != "attribute":
            continue
        if hasattr(hir_op, param.name):
            attributes[param.name] = getattr(hir_op, param.name)
        elif not param.has_default:
            return None
    return attributes


def automatic_candidate(hir_op):
    """Instantiate the sole candidate when its attributes need no choice."""
    candidates = candidate_ops(type(hir_op))
    if len(candidates) != 1:
        return None
    op_type = candidates[0]
    attributes = candidate_attributes(hir_op, op_type)
    return None if attributes is None else op_type(**attributes)


def _auto_import(pkg_name: str) -> None:
    package = importlib.import_module(pkg_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, f"{pkg_name}."):
        importlib.import_module(module_name)


_auto_import(__name__)

__all__ = [
    "automatic_candidate",
    "candidate_attributes",
    "candidate_ops",
    "register_candidates",
]
