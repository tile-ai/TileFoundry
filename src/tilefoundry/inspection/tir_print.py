"""Compact text rendering for the dynamic-shape TIR ops.

There is no full TIR pretty-printer in :mod:`tilefoundry.inspection` yet —
``python_printer`` only covers HIR. This module provides small
single-purpose formatters used by viewers / debug dumps for the three
new TIR pieces (``Abort`` / ``ShapeOf`` / ``DispatchCall``). The output
is not parser input — it is a human-readable trace.
"""
from __future__ import annotations

from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import Abort, Evaluate


def format_expr(expr) -> str:
    if isinstance(expr, ShapeOf):
        return f"shape_of({expr.param.name}, {expr.axis})"
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Constant):
        return repr(expr.value)
    return repr(expr)


def format_pattern(pat: Pattern) -> str:
    if isinstance(pat, DimVarRangePat):
        return f"{pat.dim_var} in [{pat.lo},{pat.hi}]"
    return repr(pat)


def format_symbol_call(call: Evaluate) -> str:
    callee_name = call.callable.name
    args = ", ".join(format_expr(a) for a in call.args)
    return f"{callee_name}({args})"


def format_abort(stmt: Abort) -> str:
    if stmt.message:
        return f'abort({stmt.message!r})'
    return "abort()"


def format_dispatch_call(stmt: DispatchCall, indent: str = "") -> str:
    """Compact multi-line rendering for ``DispatchCall``.

    Header line ``dispatch <callee_name>:`` followed by one indented
    line per case (in source order) and a final ``fallback`` line.
    Preserving source order satisfies the IR contract.
    """
    lines = [f"{indent}dispatch {stmt.callee_name}:"]
    for pats, call in zip(stmt.case_patterns, stmt.case_calls):
        pat_str = ", ".join(format_pattern(p) for p in pats)
        lines.append(f"{indent}  case {pat_str}: {format_symbol_call(call)}")


    fb_body = stmt.fallback.body
    if len(fb_body) == 1 and isinstance(fb_body[0], Abort):
        lines.append(f"{indent}  fallback: {format_abort(fb_body[0])}")
    else:
        lines.append(f"{indent}  fallback: <{len(fb_body)} stmts>")
    return "\n".join(lines)


__all__ = [
    "format_expr",
    "format_pattern",
    "format_symbol_call",
    "format_abort",
    "format_dispatch_call",
]
