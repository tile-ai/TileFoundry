"""Expose the author-facing ``tf`` and ``T`` dialect namespaces.

Registry-backed modules resolve operation names lazily; parser-owned tensor
annotations and string dtype sugar complete the source surface. Generated
stubs provide static completion.

See [parser §2](docs/spec/parser.md#2-syntax-and-rules).
"""

from __future__ import annotations

# ruff: noqa: I001 -- curated re-export order; alphabetical sort breaks staged imports.

from tilefoundry.dsl import tf, T
from tilefoundry.dsl._tensor import ConstTensor, Tensor


from tilefoundry.script import func, prim_func
from tilefoundry.module import module
from tilefoundry.ir import types as _types
from tilefoundry.ir.types import *
from tilefoundry.ir.pattern import RangePattern, Pattern
from tilefoundry.ir.types.dim import DimVar, ceildiv
from tilefoundry.ir.core.kinds import ReduceKind, UnaryKind, BinaryKind

__all__ = [
    "tf",
    "T",
    "ConstTensor",
    "Tensor",
    "func",
    "module",
    "prim_func",
    *_types.__all__,
    "Pattern",
    "RangePattern",
    "DimVar",
    "ceildiv",
    "ReduceKind",
    "UnaryKind",
    "BinaryKind",
]
