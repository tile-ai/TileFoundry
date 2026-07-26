"""TileFoundry runtime layer — the ``RuntimeModule`` twin of an ir ``Module``
(authored with ``@runtime_module`` / ``@runtime_func``), the
``RuntimeFunction`` implementation base, the compiled-path ``CompiledModule``,
checkpoint ``RuntimeResource``s, and ``check`` / ``bench``. See
docs/spec/runtime.md §1.
"""
from __future__ import annotations

from .decorator import runtime_func, runtime_module
from .function import (
    EntryABI,
    ParamABI,
    RuntimeFunction,
    entry_abi_of,
    param_abi_of,
)
from .measure import Gate, Report, bench, check
from .module import CompiledModule, RuntimeModule
from .resource import DictResource, RuntimeResource, SafetensorsResource

__all__ = [
    "CompiledModule",
    "DictResource",
    "EntryABI",
    "Gate",
    "ParamABI",
    "Report",
    "RuntimeFunction",
    "RuntimeModule",
    "RuntimeResource",
    "SafetensorsResource",
    "bench",
    "check",
    "entry_abi_of",
    "param_abi_of",
    "runtime_func",
    "runtime_module",
]
