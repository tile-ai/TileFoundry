"""TileFoundry runtime layer — the ``RuntimeModule`` twin of an ir ``Module``
(authored with ``@runtime_module`` / ``@runtime_func``), the
``RuntimeFunction`` implementation base, the compiled-path ``CompiledModule``,
checkpoint ``RuntimeResource``s, and ``check``. See
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
from .measure import (
    PREDICATES,
    AllClose,
    Cosine,
    Equal,
    MaxAbs,
    MaxRel,
    NanInf,
    OutputCheck,
    Predicate,
    PredicateResult,
    RelL2,
    Report,
    Ulp,
    check,
)
from .module import CompiledModule, RuntimeModule
from .resource import Absolute, DictResource, Preprocessed, RuntimeResource, SafetensorsResource

__all__ = [
    "PREDICATES",
    "Absolute",
    "AllClose",
    "CompiledModule",
    "Cosine",
    "DictResource",
    "EntryABI",
    "Equal",
    "MaxAbs",
    "MaxRel",
    "NanInf",
    "OutputCheck",
    "ParamABI",
    "Predicate",
    "PredicateResult",
    "Preprocessed",
    "RelL2",
    "Report",
    "RuntimeFunction",
    "RuntimeModule",
    "RuntimeResource",
    "SafetensorsResource",
    "Ulp",
    "check",
    "entry_abi_of",
    "param_abi_of",
    "runtime_func",
    "runtime_module",
]
