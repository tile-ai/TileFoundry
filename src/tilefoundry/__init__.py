"""TileFoundry top-level package.

Re-exports the stable public API from `tilefoundry.ir.*` for convenience.
Spec 011 §1 is authoritative on physical layout.
"""

from __future__ import annotations

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

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
    costmodel_registry,
    lower_cpu_registry,
    lower_cuda_registry,
    register_costmodel,
    register_lower_cpu,
    register_lower_cuda,
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
    LayoutLike,
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

# TupleGetItem moved from core.expr to hir.tensor as a proper Op.
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem

# Spec 000 / 006 public surface: @tilefoundry.func / @tilefoundry.prim_func / intrinsic.
from tilefoundry.script import func, intrinsic, prim_func
from tilefoundry.module import module

# Top-level pipeline entry.
from tilefoundry.compile import build, compile, jit, lower, normalize_to_module, CompilerOptions
from tilefoundry.inspection.viewer import Viewer as _Viewer

# All imports done — now invoke the deferred dim typeinfer registration.
_register_dim_typeinfer()


def view(root, *, port: int = 0, open_browser: bool = True) -> int:
    """Start the interactive HIR viewer for *root* (Function or Module).

    Thin wrapper around ``tilefoundry.inspection.viewer.Viewer(root).serve``.
    """
    return _Viewer(root).serve(port=port, open_browser=open_browser)

__all__ = [
    # core
    "Expr", "Var", "Constant", "Call", "Stmt", "TupleGetItem",
    "Op", "ParameterInfo",
    "AnalysisRegistry",
    "typeinfer_registry", "verify_stmt_registry", "costmodel_registry",
    "lower_cuda_registry", "lower_cpu_registry",
    "register_typeinfer", "register_verify_stmt", "register_costmodel",
    "register_lower_cuda", "register_lower_cpu",
    "TypeInferContext",
    "VerifyError",
    # types
    "DType", "TensorType", "TupleType", "Type",
    "Pattern", "DimVarRangePat", "DimVar",
    # shard
    "IntTuple", "LayoutBase", "Layout", "ComposedLayout", "LayoutLike",
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
