"""TileFoundry runtime layer — ``RuntimeModule`` container, ABI, checkpoint
resource, and measurement helpers.

Two construction paths for a ``RuntimeModule``:
- compiled: ``tilefoundry.build`` / ``compile`` / ``jit`` → codegen →
  ``LinkedModule`` → ``runtime.loader.load_linked_module``.
- handwritten: direct construction — wrap a plain torch/triton callable in a
  ``RuntimeFunction`` and assemble a ``RuntimeModule`` by hand, resolving
  weights/states from a ``RuntimeResource`` (e.g. ``SafetensorsResource``).
"""
from __future__ import annotations

from .function import (
    CallableType,
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
