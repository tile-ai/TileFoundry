"""TileFoundry top-level package.

Re-exports the stable public API from `tilefoundry.ir.*` for convenience.
[code-organization §1](docs/spec/code-organization.md#1-directory-skeleton) is
authoritative on physical layout.
"""

from __future__ import annotations

from importlib.metadata import version

from . import bootstrap
from .compiler import Options as CompilerOptions
from .compiler import build, compile, jit, normalize_to_module
from .core import ShardTensor
from .inspection.viewer import view
from .ir import pattern
from .ir.core import Stmt
from .ir.expr import (
    Call,
    Constant,
    DimVar,
    DType,
    Expr,
    Op,
    ParameterInfo,
    Var,
    VerifyError,
)
from .ir.hir.tensor import TupleGetItem
from .ir.types import (
    B,
    Broadcast,
    ComposedLayout,
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
    Swizzle,
    TensorType,
    Topology,
    TupleType,
    Type,
    V,
    Varying,
)
from .parser import func, intrinsic, module, prim_func

__version__ = version("tilefoundry")

__all__ = [
    "__version__",
    "bootstrap",
    "ShardTensor",
    "Expr",
    "Var",
    "Constant",
    "Call",
    "Stmt",
    "TupleGetItem",
    "Op",
    "ParameterInfo",
    "VerifyError",
    "DType",
    "TensorType",
    "TupleType",
    "Type",
    "pattern",
    "DimVar",
    "IntTuple",
    "LayoutBase",
    "Layout",
    "Swizzle",
    "ComposedLayout",
    "Topology",
    "Mesh",
    "ShardAttr",
    "Split",
    "Partial",
    "Broadcast",
    "V",
    "Varying",
    "ShardLayout",
    "S",
    "P",
    "B",
    "func",
    "prim_func",
    "intrinsic",
    "module",
    "build",
    "compile",
    "jit",
    "normalize_to_module",
    "CompilerOptions",
    "view",
]
