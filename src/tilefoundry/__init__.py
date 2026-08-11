"""TileFoundry top-level package.

Re-exports the stable public API from `tilefoundry.ir.*` for convenience.
[code-organization §1](docs/spec/code-organization.md#1-directory-skeleton) is
authoritative on physical layout.
"""

from __future__ import annotations

from importlib.metadata import version as _distribution_version

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

__version__ = _distribution_version("tilefoundry")


from tilefoundry.ir.core import (
    AnalysisRegistry,
    Call,
    Constant,
    Expr,
    FunctionScope,
    Op,
    ParameterInfo,
    TypeInferContext,
    Var,
    VerifyError,
    cost_evaluator_registry,
    register_cost_evaluator,
    register_typeinfer,
    register_verify_stmt,
    typeinfer_registry,
    verify_stmt_registry,
)


from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.types import DType, TensorType, TupleType, Type
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import (
    B,
    Broadcast,
    ComposedLayout,
    Dynamic,
    IntTuple,
    Layout,
    LayoutBase,
    Mesh,
    P,
    Partial,
    S,
    ShardAttr,
    ShardLayout,
    Split,
    Topology,
)


from tilefoundry.ir.tir.stmt import Stmt




from tilefoundry.ir.types import _register_dim_typeinfer


from tilefoundry.ir import hir as _hir  # noqa: F401
from tilefoundry.ir import tir as _tir  # noqa: F401




from tilefoundry.visitor_registry import op_cost as _op_cost  # noqa: F401


from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem


from tilefoundry.script import func, intrinsic, prim_func
from tilefoundry.module import module


from tilefoundry.compile import build, compile, jit, lower, normalize_to_module, CompilerOptions
from tilefoundry.inspection.viewer import Viewer as _Viewer


_register_dim_typeinfer()

def view(root, *, port: int = 0, open_browser: bool = True) -> int:
    """Start the interactive HIR viewer for *root* (Function or Module).

    Thin wrapper around ``tilefoundry.inspection.viewer.Viewer(root).serve``.
    """
    return _Viewer(root).serve(port=port, open_browser=open_browser)

__all__ = [
    "__version__",

    "Expr", "Var", "Constant", "Call", "Stmt", "TupleGetItem",
    "Op", "ParameterInfo",
    "AnalysisRegistry",
    "typeinfer_registry", "verify_stmt_registry", "cost_evaluator_registry",
    "register_typeinfer", "register_verify_stmt", "register_cost_evaluator",
    "TypeInferContext", "FunctionScope",
    "VerifyError",

    "DType", "TensorType", "TupleType", "Type",
    "Pattern", "DimVarRangePat", "DimVar",

    "IntTuple", "LayoutBase", "Layout", "ComposedLayout",
    "Topology", "Mesh",
    "ShardAttr", "Split", "Partial", "Broadcast", "Dynamic", "ShardLayout",
    "S", "P", "B",

    "func", "prim_func", "intrinsic", "module",

    "lower", "build", "compile", "jit",
    "normalize_to_module", "CompilerOptions",
    "view",
]
