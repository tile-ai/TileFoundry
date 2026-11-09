"""DSL parsing: the authored Python function and module bodies to IR."""

from .ast_pattern import (
    FuncContext,
    FunctionRole,
    LexicalScope,
    ParseError,
    consume_module_context,
    create_module_context,
    module_context_for_frame,
    target_hardware_context,
)
from .module_decorator import module
from .parser_visitor import parse_function
from .script import func, intrinsic, prim_func
