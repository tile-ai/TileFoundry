from __future__ import annotations

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

from .errors import VerifyError
from .expr import Call, Constant, Expr, Tuple, Var
from .metadata import (
    BindingMetadata,
    IRMetadata,
    SourceSpanMetadata,
    binding_name,
    diagnostic_location,
    get_metadata,
    remove_metadata,
    replace_metadata,
    source_metadata,
)
from .op import Op, ParameterInfo
from tilefoundry.visitor_registry.registries import (
    AnalysisRegistry,
    cost_evaluator_registry,
    register_cost_evaluator,
    register_typeinfer,
    register_verify_stmt,
    typeinfer_registry,
    verify_stmt_registry,
)
from .context import TypeInferContext

__all__ = [
    # exprs
    "Expr", "Var", "Constant", "Call", "Tuple",
    # metadata
    "IRMetadata", "BindingMetadata", "SourceSpanMetadata",
    "binding_name", "diagnostic_location", "source_metadata",
    "get_metadata", "replace_metadata", "remove_metadata",
    # op
    "Op", "ParameterInfo",
    # registry
    "AnalysisRegistry",
    "typeinfer_registry", "verify_stmt_registry", "cost_evaluator_registry",
    "register_typeinfer", "register_verify_stmt", "register_cost_evaluator",
    # context
    "TypeInferContext",
    # error
    "VerifyError",
]
