from __future__ import annotations

import importlib
import pkgutil

from .async_copy import CopyAsync, CpAsyncCommit, CpAsyncWait
from .dispatch import DispatchCall
from .launch import Launch
from .prim_function import PrimFunction
from .shape import ShapeOf, shape_var_name
from .stmts import (
    Abort,
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from .symbol_ref import SymbolRef
from .sync import Sync, SyncBarrier, classify, participation


def _auto_import(pkg_name: str) -> None:
    pkg = importlib.import_module(pkg_name)
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
        importlib.import_module(modname)


# Recursive walk so every submodule (including root-level arith / clamp /
# reduce) is imported for its registration side effects, per
# docs/spec/visitor-registry.md.
_auto_import("tilefoundry.ir.tir")

__all__ = [
    "PrimFunction",
    "For", "While", "If", "MeshScope", "Return",
    "Sequential", "LetStmt",
    "Abort", "DispatchCall", "Launch", "ShapeOf", "shape_var_name",
    "SymbolRef", "Sync", "SyncBarrier", "classify", "participation",
    "CopyAsync", "CpAsyncCommit", "CpAsyncWait",
]
