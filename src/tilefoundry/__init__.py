"""TileFoundry top-level package.

Re-exports the stable public API from `tilefoundry.ir.*` for convenience.
[code-organization §1](docs/spec/code-organization.md#1-directory-skeleton) is authoritative on physical layout.
"""

from __future__ import annotations

from importlib.metadata import version as _distribution_version

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

__version__ = _distribution_version("tilefoundry")

# Core IR
from tilefoundry.ir.core import (
    AnalysisRegistry,
    Call,
    Constant,
    Expr,
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

# Type system
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
    MeshAxis,
    P,
    Partial,
    S,
    ShardAttr,
    ShardLayout,
    Split,
    Topology,
)

# Tir (Stmt base + PrimFunction)
from tilefoundry.ir.tir.stmt import Stmt

# dim.* typeinfer can't run at types/__init__ time because of an
# import cycle, so it's exposed as ``_register_dim_typeinfer`` and
# invoked once at the end of this module after the public imports.
from tilefoundry.ir.types import _register_dim_typeinfer

# hir and tir packages have side-effect imports (register typeinfer / verify-stmt)
from tilefoundry.ir import hir as _hir  # noqa: F401
from tilefoundry.ir import tir as _tir  # noqa: F401

# Every operation's per-instance work, registered once for whoever asks: the
# analysis layer costs a program with it and the scheduling algorithms price
# their candidates with it. Imported after the ops themselves exist.
from tilefoundry.visitor_registry import op_cost as _op_cost  # noqa: F401

# TupleGetItem moved from core.expr to hir.tensor as a proper Op.
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem

# Spec 000 / 006 public surface: @tilefoundry.func / @tilefoundry.prim_func / intrinsic.
from tilefoundry.script import func, intrinsic, prim_func
from tilefoundry.module import module

# Top-level pipeline entry.
from tilefoundry.compile import build, compile, jit, lower, normalize_to_module, CompilerOptions
from tilefoundry.inspection.viewer import Viewer as _Viewer
from tilefoundry.target import register_facts_projections as _register_facts_projections
from tilefoundry.target import register_schedule_algorithms as _register_schedule_algorithms

# All imports done — now invoke the deferred dim typeinfer registration.
_register_dim_typeinfer()

# Deferred for the same reason: a Target's Facts projections name the analysis
# aggregates they build, and the analysis layer rests on the IR that was still
# loading the Target while this module ran.
_register_facts_projections()
_register_schedule_algorithms()


def view(root, *, port: int = 0, open_browser: bool = True) -> int:
    """Start the interactive HIR viewer for *root* (Function or Module).

    Thin wrapper around ``tilefoundry.inspection.viewer.Viewer(root).serve``.
    """
    return _Viewer(root).serve(port=port, open_browser=open_browser)

__all__ = [
    "__version__",
    # core
    "Expr", "Var", "Constant", "Call", "Stmt", "TupleGetItem",
    "Op", "ParameterInfo",
    "AnalysisRegistry",
    "typeinfer_registry", "verify_stmt_registry", "cost_evaluator_registry",
    "register_typeinfer", "register_verify_stmt", "register_cost_evaluator",
    "TypeInferContext",
    "VerifyError",
    # types
    "DType", "TensorType", "TupleType", "Type",
    "Pattern", "DimVarRangePat", "DimVar",
    # shard
    "IntTuple", "LayoutBase", "Layout", "ComposedLayout",
    "Topology", "MeshAxis", "Mesh",
    "ShardAttr", "Split", "Partial", "Broadcast", "Dynamic", "ShardLayout",
    "S", "P", "B",
    # public decorator surface
    "func", "prim_func", "intrinsic", "module",
    # pipeline entry
    "lower", "build", "compile", "jit",
    "normalize_to_module", "CompilerOptions",
    "view",
]
