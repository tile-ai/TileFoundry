"""TileFoundry runtime layer — the ``RuntimeModule`` twin of an ir ``Module``,
the ``RuntimeFunction`` implementation base, checkpoint ``RuntimeResource``s,
and ``check`` / ``bench``. See docs/spec/runtime.md §1 for the contract.
"""
from __future__ import annotations

from .function import (
    CallableType,
    CompiledFunction,
    KernelInfo,
    LaunchConfig,
    ParamABI,
    RuntimeFunction,
    callable_type_of,
)
from .measure import Gate, Report, bench, check
from .module import RuntimeModule
from .resource import DictResource, RuntimeResource, SafetensorsResource

__all__ = [
    "CallableType",
    "CompiledFunction",
    "DictResource",
    "Gate",
    "KernelInfo",
    "LaunchConfig",
    "ParamABI",
    "Report",
    "RuntimeFunction",
    "RuntimeModule",
    "RuntimeResource",
    "SafetensorsResource",
    "bench",
    "callable_type_of",
    "check",
]
