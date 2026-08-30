"""Function parser entry points for the AST pattern prototype."""

from __future__ import annotations

import ast
import dataclasses
import inspect
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
        initial_mesh_depth = len(self.context.state.mesh_stack)
        context = MatchContext.from_function(self.context)
        try:
            return parse_node(self.root_pattern, node, context)
        finally:
            del self.context.state.mesh_stack[initial_mesh_depth:]

    def visit_function(self, node: ast.FunctionDef) -> Any:
        return self.visit(node)


def _extract_function_def(fn: FunctionType) -> tuple[ast.FunctionDef, str]:
    if not isinstance(fn, FunctionType):
        raise TypeError(f"parse_function expects a Python function, got {type(fn).__name__}")
    try:
        source_lines, start_line = inspect.getsourcelines(fn)
    except (OSError, TypeError) as error:
        raise TypeError("parse_function requires authored source for the function") from error
    source_filename = inspect.getsourcefile(fn) or "<string>"
    source = "".join(source_lines)
    try:
        module = ast.parse(source, filename=source_filename)
        source_start_line = 1
    except IndentationError:
        module = ast.parse(f"if True:\n{source}", filename=source_filename)
        source_start_line = 2
    ast.increment_lineno(module, start_line - source_start_line)
    functions = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise TypeError("parse_function requires exactly one authored FunctionDef")
    return functions[0], source_filename


def parse_function(fn: FunctionType, context: FuncParserContext) -> Any:
    """Parse one authored Python function using its typed parser context."""
    node, source_filename = _extract_function_def(fn)
    context = dataclasses.replace(
        context,
        source_filename=(
            source_filename if context.source_filename == "<string>" else context.source_filename
        ),
    )
    return FuncParserVisitor(context).visit_function(node)


__all__ = [
    "FuncParserVisitor",
    "parse_function",
]
