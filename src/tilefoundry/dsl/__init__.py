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


from tilefoundry.script import func
from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.types.dim import DimVar, ceildiv
from tilefoundry.ir.types.shard import (
    Mesh,
    Topology,
    Split,
    Partial,
    Broadcast,
    S,
    P,
    B,
)
from tilefoundry.ir.core.kinds import ReduceKind, UnaryKind, BinaryKind

__all__ = [
    "tf",
    "T",
    "ConstTensor",
    "Tensor",
    "func",
    "Pattern",
    "DimVarRangePat",
    "DimVar",
    "ceildiv",
    "Mesh",
    "Topology",
    "Split",
    "Partial",
    "Broadcast",
    "S",
    "P",
    "B",
    "ReduceKind",
    "UnaryKind",
    "BinaryKind",
]
