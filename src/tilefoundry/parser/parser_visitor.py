"""Function parser entry points for the AST pattern prototype."""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import FunctionType
from typing import Any

from .ast_pattern import (
    FuncParserContext,
    FunctionPattern,
    MatchContext,
    parse_node,
)


class FuncParserVisitor:
    """Walk one authored function from its selected root pattern."""

    def __init__(self, context: FuncParserContext):
        self.context = context
        self.root_pattern = FunctionPattern()

    def visit(self, node: ast.AST) -> Any:
        return parse_node(self.root_pattern, node, MatchContext.from_function(self.context))

    def visit_function(self, node: ast.FunctionDef) -> Any:
        return self.visit(node)


def _extract_function_def(fn: FunctionType) -> ast.FunctionDef:
    if not isinstance(fn, FunctionType):
        raise TypeError(f"parse_function expects a Python function, got {type(fn).__name__}")
    try:
        source_lines, start_line = inspect.getsourcelines(fn)
        source = textwrap.dedent("".join(source_lines))
    except (OSError, TypeError) as error:
        raise TypeError("parse_function requires authored source for the function") from error
    module = ast.parse(source, filename=inspect.getsourcefile(fn) or "<string>")
    ast.increment_lineno(module, start_line - 1)
    functions = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise TypeError("parse_function requires exactly one authored FunctionDef")
    return functions[0]


def parse_function(fn: FunctionType, context: FuncParserContext) -> Any:
    """Parse one authored Python function using its typed parser context."""
    return FuncParserVisitor(context).visit_function(_extract_function_def(fn))


__all__ = [
    "FuncParserVisitor",
    "parse_function",
]
