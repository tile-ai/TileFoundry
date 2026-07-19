from __future__ import annotations

from .hir_parser import parse_func
from .tir_parser import parse_prim_func

__all__ = ["parse_func", "parse_prim_func"]
