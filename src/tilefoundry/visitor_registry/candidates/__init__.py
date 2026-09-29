"""Registered HIR-to-TIR candidate pairings."""

from __future__ import annotations

import importlib
import pkgutil

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


def _auto_import(pkg_name: str) -> None:
    package = importlib.import_module(pkg_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, f"{pkg_name}."):
        importlib.import_module(module_name)


_auto_import(__name__)

__all__ = ["candidate_ops", "register_candidates"]
