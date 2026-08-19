from __future__ import annotations

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

from .errors import VerifyError
from .expr import Call, Constant, Expr, Tuple, Var
from .metadata import (
    BindingMetadata,
    ExecutionDomainMetadata,
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
from .values import TotalAndPerUnit, TripInterval
from tilefoundry.visitor_registry.registries import (
    AnalysisRegistry,
    cost_evaluator_registry,
    register_cost_evaluator,
    register_typeinfer,
    register_verify_stmt,
    typeinfer_registry,
    verify_stmt_registry,
)
from .context import FunctionScope, TypeInferContext

__all__ = [
    "Expr",
    "Var",
    "Constant",
    "Call",
    "Tuple",
    "IRMetadata",
    "BindingMetadata",
    "ExecutionDomainMetadata",
    "SourceSpanMetadata",
    "binding_name",
    "diagnostic_location",
    "source_metadata",
    "get_metadata",
    "replace_metadata",
    "remove_metadata",
    "Op",
    "ParameterInfo",
    "TotalAndPerUnit",
    "TripInterval",
    "AnalysisRegistry",
    "typeinfer_registry",
    "verify_stmt_registry",
    "cost_evaluator_registry",
    "register_typeinfer",
    "register_verify_stmt",
    "register_cost_evaluator",
    "FunctionScope",
    "TypeInferContext",
    "VerifyError",
]
