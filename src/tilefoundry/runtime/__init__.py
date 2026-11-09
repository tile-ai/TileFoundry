"""TileFoundry runtime layer.

TileFoundry runtime layer — the ``runtime.Module`` twin of an ir ``Module``
(authored with ``@runtime.module`` / ``@runtime.func``), the
``runtime.Function`` implementation base, the compiled-path ``CompiledModule``,
checkpoint ``runtime.Resource``s, and ``check``. See
[runtime §1](docs/spec/runtime.md#1-python-runtime-surface).
"""
from __future__ import annotations

from .decorator import func, module
from .function import Function
from .loader import load_linked_module
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
    flatten_outputs,
)
from .module import CompiledModule, Module
from .prepare import prepare, prepare_into
from .resource import (
    Absolute,
    DictResource,
    DrawnResource,
    Preprocessed,
    Resource,
    SafetensorsResource,
    draw_tensor,
    resolved_device,
)
