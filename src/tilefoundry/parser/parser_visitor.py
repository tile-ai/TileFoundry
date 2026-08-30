"""Function parser entry points for the AST pattern prototype."""

from __future__ import annotations

import ast
import dataclasses
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
        initial_mesh_depth = len(self.context.state.mesh_stack)
        context = MatchContext.from_function(self.context)
        try:
            return parse_node(self.root_pattern, node, context)
        finally:
            del self.context.state.mesh_stack[initial_mesh_depth:]

    def visit_function(self, node: ast.FunctionDef) -> Any:
        return self.visit(node)


def _dedented_source(source_lines: list[str]) -> tuple[str, int]:
    """Return parseable source and the common source-file indentation removed."""
    original = "".join(source_lines)
    dedented = textwrap.dedent(original)
    for original_line, dedented_line in zip(original.splitlines(), dedented.splitlines()):
        if not dedented_line:
            continue
        if original_line.endswith(dedented_line):
            prefix = original_line[: len(original_line) - len(dedented_line)]
            if not prefix or prefix.isspace():
                return dedented, len(prefix)
        break
    return dedented, 0


def _extract_function_def(fn: FunctionType) -> tuple[ast.FunctionDef, int, str]:
    if not isinstance(fn, FunctionType):
        raise TypeError(f"parse_function expects a Python function, got {type(fn).__name__}")
    try:
        source_lines, start_line = inspect.getsourcelines(fn)
        source, source_column_offset = _dedented_source(source_lines)
    except (OSError, TypeError) as error:
        raise TypeError("parse_function requires authored source for the function") from error
    source_filename = inspect.getsourcefile(fn) or "<string>"
    module = ast.parse(source, filename=source_filename)
    ast.increment_lineno(module, start_line - 1)
    functions = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise TypeError("parse_function requires exactly one authored FunctionDef")
    return functions[0], source_column_offset, source_filename


def parse_function(fn: FunctionType, context: FuncParserContext) -> Any:
    """Parse one authored Python function using its typed parser context."""
    node, source_column_offset, source_filename = _extract_function_def(fn)
    context = dataclasses.replace(
        context,
        source_filename=(
            source_filename if context.source_filename == "<string>" else context.source_filename
        ),
        source_column_offset=source_column_offset,
    )
    return FuncParserVisitor(context).visit_function(node)


__all__ = [
    "FuncParserVisitor",
    "parse_function",
]
