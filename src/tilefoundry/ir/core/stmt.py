"""Shared ``Stmt`` base without an ``ir.tir`` dependency.

Concrete statements live in ``ir.tir.stmts``. Keeping the base in core avoids
an import cycle with visitor contexts; ``ir.tir.stmt`` re-exports it for
compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stmt:
    """tir-only. hir does not contain Stmt nodes.

    Structural Stmts (Sequential / LetStmt / For / While / If /
    MeshScope / Return / Evaluate / PrimFunction) are not part of any
    callable registry; effect-ful TIR Ops register themselves via
    ``@register_op`` (and live in Stmt position via
    ``Evaluate(op, args)``).
    """

    loc: str | None = field(default=None, kw_only=True)
