"""DSL-name → Op / Stmt dispatch.

This module is a thin facade over ``tilefoundry.ir.core.op_registry``:

- ``resolve_op(name)`` / ``resolve_stmt(name)`` — flat-name lookup into
  the dialect-specific registry.
- ``resolve_callable(name, token)`` — dialect-strict dispatch:
  HIR body resolves only HIR Ops; TIR body resolves only TIR Stmts +
  user-registered intrinsics. **No cross-dialect fallback** (TIR body
  must use the future ``tf.<category>.<name>`` namespace if it really
  wants to embed an HIR Op as a value).
- ``_binary_kind_for_ast_op(ast_op)`` / ``_unary_kind_for_ast_op(ast_op)``
  — Python AST binop / unaryop → ``BinaryKind`` / ``UnaryKind``. Parser
  uses these to construct ``Binary`` / ``Unary`` directly without
  routing through a per-name HIR Op class (the 19 legacy sugar Op
  classes collapse into alias schemas + tag-dispatch IR).
Every callable Op self-registers via the ``@register_op``
decorator; ``resolve_op`` / ``resolve_stmt`` look the class up through
the OpSchema list-per-name registry. The legacy ``_hir_*`` / ``_tir_*``
static dictionaries and the metaclass-driven auto-register are gone.
"""
from __future__ import annotations

from typing import Literal

from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.op_registry import (
    _first_schema,
    get_op_by_name,
    get_stmt_by_name,
)
from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.tir.intrinsic import _intrinsic_dispatch

Token = Literal["hir", "tir"]


# AST-level Python operators map directly to Binary / Unary
# kinds without going through a per-name Op class. The parser
# constructs ``Binary(kind=...)`` / ``Unary(kind=...)`` directly
# instead of resolving a callable name and re-binding.

def _binary_kind_for_ast_op(ast_op_name: str):
    _MAP = {
        "Add": BinaryKind.ADD, "Sub": BinaryKind.SUB,
        "Mult": BinaryKind.MUL, "Div": BinaryKind.DIV,
        "FloorDiv": BinaryKind.FLOOR_DIV, "Mod": BinaryKind.MOD,
        "Eq": BinaryKind.EQ, "NotEq": BinaryKind.NE,
        "Lt": BinaryKind.LT, "LtE": BinaryKind.LE,
        "Gt": BinaryKind.GT, "GtE": BinaryKind.GE,
        "And": BinaryKind.AND, "Or": BinaryKind.OR,
    }
    return _MAP.get(ast_op_name)


def _unary_kind_for_ast_op(ast_op_name: str):
    _MAP = {"USub": UnaryKind.NEG, "Not": UnaryKind.NOT}
    return _MAP.get(ast_op_name)


def resolve_op(name: str) -> type | None:
    """Resolve a DSL bare-call name to an HIR Op subclass, or ``None``.

    Skips alias schemas (``op_class=None``); this returns the concrete
    legacy class. Use :func:`resolve_schema` to honour aliases.
    """
    return get_op_by_name(name)


def resolve_schema(name: str, dialect: str = "tf") -> OpSchema | None:
    """Resolve a DSL bare-call name to its first ``OpSchema`` (alias-aware).

    A surface name may map to a surface-alias schema
    (``schema.op_class is None``) prepended over a legacy real-Op
    schema. Parser dispatch uses this resolver so the alias wins
    first-match — its ``builder`` constructs the kinded target Op
    (e.g. ``Binary(kind=ADD)``) instead of the legacy class.
    """
    return _first_schema(dialect, name)


def resolve_stmt(name: str) -> type | None:
    """Resolve a DSL bare-call name to a TIR Stmt subclass, or ``None``.

    Falls through to user-registered intrinsics (``@intrinsic`` decorator)
    so user-defined effect Stmts continue to participate in TIR dispatch
    without going through the canonical opt-in registry.
    """
    cls = get_stmt_by_name(name)
    if cls is not None:
        return cls
    return _intrinsic_dispatch.get(name)


def resolve_callable(name: str, token: Token) -> tuple[str, type]:
    """Dispatch *name* in the given DSL *token* dialect.

    Returns ``("op" | "stmt", cls)`` or raises ``VerifyError``.

    Trailing-underscore convention ([parser §1.3](docs/spec/parser.md#13-op-call)): ``foo_`` is an
    explicit effect-form selector.

    Strict dialect routing (no cross-dialect fallback):
    - ``hir`` body resolves only HIR Ops.
    - ``tir`` body resolves only TIR Stmts (plus user intrinsics).
      An HIR Op embedded in a TIR body must use the future
      ``tf.<category>.<name>`` namespace surface (authoring-namespace
      plan). Bare-name HIR Op fallback in TIR is intentionally removed.
    """
    if token == "tir":
        # Trailing-underscore effect-form selector is TIR-only: leaving it
        # unconditional was an HIR escape hatch into the TIR Stmt registry,
        # violating strict dialect routing.
        if name.endswith("_") and not name.startswith("_"):
            base = name[:-1]
            stmt = resolve_stmt(base)
            if stmt is not None:
                return ("stmt", stmt)
        stmt = resolve_stmt(name)
        if stmt is not None:
            return ("stmt", stmt)
        raise VerifyError(
            f"unknown TIR callable {name!r} in @tilefoundry.prim_func body "
            f"(bare HIR Op fallback removed; use tf.<category>.<name> "
            f"namespace if this is meant to be an HIR Op)"
        )
    op = resolve_op(name)
    if op is not None:
        return ("op", op)
    raise VerifyError(f"unknown HIR callable {name!r} in @tilefoundry.func body")


# Legacy ``binop_to_op_cls`` removed — AST-level Python
# operator dispatch now uses :func:`_binary_kind_for_ast_op` /
# :func:`_unary_kind_for_ast_op` and constructs ``Binary`` / ``Unary``
# directly. ``rebind_to_kinded`` is gone for the same reason: the 19
# legacy sugar Op classes that it routed are deleted, and surface
# names route through ``@register_alias`` schemas.


__all__ = [
    "resolve_op",
    "resolve_schema",
    "resolve_stmt",
    "resolve_callable",
    "_binary_kind_for_ast_op",
    "_unary_kind_for_ast_op",
]
