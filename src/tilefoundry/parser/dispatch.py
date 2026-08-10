"""Resolve DSL names and Python operators through dialect registries.

HIR resolves only HIR operations; TIR resolves only TIR statements and user
intrinsics, with no cross-dialect fallback. Binary and unary AST operators map
directly to kinded IR. Registered schemas provide flat-name operation and
statement lookup.
See [parser §4.6](docs/spec/parser.md#46-per-dialect-strict-resolution).
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
    """Dispatch *name* within one strict DSL dialect.

    Return an operation or statement kind and class, or raise ``VerifyError``.
    A trailing underscore explicitly selects effect form in TIR. HIR names never
    fall back to TIR, and TIR names never fall back to HIR.
    See [parser §1.3](docs/spec/parser.md#13-op-call).
    """
    if token == "tir":



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










__all__ = [
    "resolve_op",
    "resolve_schema",
    "resolve_stmt",
    "resolve_callable",
    "_binary_kind_for_ast_op",
    "_unary_kind_for_ast_op",
]
