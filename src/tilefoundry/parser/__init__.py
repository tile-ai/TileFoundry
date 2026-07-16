from __future__ import annotations

from .hir_parser import parse_func, parse_func_source, parse_module_source, parse_script
from .tir_parser import parse_prim_func

__all__ = [
    "parse_func",
    "parse_func_source",
    "parse_module_source",
    "parse_script",
    "parse_prim_func",
]
