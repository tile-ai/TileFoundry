"""Typeinfer diagnostic discipline (hir.md: constraints enforced via
``ctx.error(...)``, never a bare ``raise TypeError``).
"""
from __future__ import annotations

import ast
import pathlib

_HIR_ROOT = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "tilefoundry" / "ir" / "hir"
)


def _is_register_typeinfer_decorator(node: ast.expr) -> bool:
    """True if *node* is (a call to) ``register_typeinfer``, however
    imported (``register_typeinfer`` / ``visitor_registry.register_typeinfer``)."""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id == "register_typeinfer"
    if isinstance(target, ast.Attribute):
        return target.attr == "register_typeinfer"
    return False


def _raises_type_error(fn: ast.FunctionDef) -> list[int]:
    lines = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        target = exc.func if isinstance(exc, ast.Call) else exc
        if isinstance(target, ast.Name) and target.id == "TypeError":
            lines.append(node.lineno)
    return lines


def _typeinfer_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(_is_register_typeinfer_decorator(d) for d in node.decorator_list):
            found.append(node)
    return found


def test_no_bare_type_error_in_typeinfer_bodies() -> None:
    violations: list[str] = []
    for path in sorted(_HIR_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for fn in _typeinfer_functions(tree):
            for lineno in _raises_type_error(fn):
                violations.append(f"{path}:{lineno} in {fn.name}()")
    assert not violations, (
        "raise TypeError inside a @register_typeinfer body (use ctx.error instead): "
        + "; ".join(violations)
    )
