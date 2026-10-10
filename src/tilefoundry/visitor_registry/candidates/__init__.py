"""Registered HIR-to-TIR candidate pairings."""

from __future__ import annotations

import importlib
import pkgutil

from tilefoundry.ir.core.op import Op
from tilefoundry.ir.core.param_def import collect_param_defs

_CANDIDATES: dict[type, list[type]] = {}
_LANDS: set[tuple[type, type]] = set()


def register_candidates(hir_op: type, tir_ops: tuple[type, ...], *, lands: bool = False) -> None:
    """Register carriers, optionally deriving their landing type from the instruction."""
    registered = _CANDIDATES.setdefault(hir_op, [])
    for tir_op in tir_ops:
        if tir_op not in registered:
            registered.append(tir_op)
        if lands:
            _LANDS.add((hir_op, tir_op))


def candidate_lands(hir_op: type, tir_op: type) -> bool:
    """Whether this pairing declares its own destination type."""
    return (hir_op, tir_op) in _LANDS


def candidate_ops(hir_op: type) -> tuple[type, ...]:
    """Return every registered TIR carrier for one HIR operation."""
    return tuple(_CANDIDATES.get(hir_op, ()))


def instruction_from_hir(hir_op: Op, op_type: type[Op]) -> Op | None:
    """Build an instruction from same-named HIR attributes, or None if incomplete."""
    attributes = {}
    for param in collect_param_defs(op_type):
        if param.kind != "attribute":
            continue
        if hasattr(hir_op, param.name):
            attributes[param.name] = getattr(hir_op, param.name)
        elif not param.has_default:
            return None
    return op_type(**attributes)


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
    "instruction_from_hir",
    "sole_candidate",
    "candidate_ops",
    "candidate_lands",
    "register_candidates",
]
