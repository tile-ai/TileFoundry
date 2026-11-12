"""Expose the author-facing ``tf`` and ``T`` dialect namespaces.

Registry-backed modules resolve operation names lazily; parser-owned tensor
annotations and string dtype sugar complete the source surface. Generated
stubs provide static completion.

See [parser §2](docs/spec/parser.md#2-syntax-and-rules).
"""

from __future__ import annotations

from tilefoundry.ir import pattern
from tilefoundry.ir.core import ModuleList, shape_of
from tilefoundry.ir.dim import ceildiv
from tilefoundry.ir.expr import (
    TARGET_MEMORY_OWNER,
    BinaryKind,
    BoolDType,
    DimVar,
    DType,
    FloatDType,
    IntegerDType,
    ReduceKind,
    StorageKind,
    StorageOwners,
    UnaryKind,
    resolve_storage,
)
from tilefoundry.ir.types import (
    B,
    Broadcast,
    CallableType,
    CommunicationHandleType,
    ComposedLayout,
    IntTuple,
    Layout,
    LayoutBase,
    Mesh,
    P,
    Partial,
    Placement,
    PointerType,
    S,
    ShardAttr,
    ShardLayout,
    Split,
    Swizzle,
    TensorType,
    Topology,
    TupleType,
    Type,
    UnitType,
    V,
    Varying,
    callable_type_for,
    local_layout,
    local_layout_and_offset,
    local_readback,
    mesh_coords,
    partial_reductions_by_axis,
    shard_layout_of,
)
from tilefoundry.ir.utils import make_shard_tensor_type
from tilefoundry.parser import func, module, prim_func

from . import T, namespace, tf
from .tensor import ConstTensor, Tensor

__all__ = [
    "CommunicationHandleType",
    "Varying",
    "V",
    "tf",
    "T",
    "namespace",
    "ConstTensor",
    "Tensor",
    "func",
    "module",
    "ModuleList",
    "prim_func",
    "shape_of",
    "TARGET_MEMORY_OWNER",
    "StorageOwners",
    "resolve_storage",
    "local_layout",
    "local_layout_and_offset",
    "local_readback",
    "mesh_coords",
    "partial_reductions_by_axis",
    "shard_layout_of",
    "B",
    "BoolDType",
    "Broadcast",
    "CallableType",
    "ComposedLayout",
    "DType",
    "FloatDType",
    "IntTuple",
    "IntegerDType",
    "Layout",
    "LayoutBase",
    "Mesh",
    "P",
    "Partial",
    "Placement",
    "PointerType",
    "S",
    "ShardAttr",
    "ShardLayout",
    "Split",
    "StorageKind",
    "Swizzle",
    "TensorType",
    "Topology",
    "TupleType",
    "Type",
    "UnitType",
    "callable_type_for",
    "make_shard_tensor_type",
    "pattern",
    "DimVar",
    "ceildiv",
    "ReduceKind",
    "UnaryKind",
    "BinaryKind",
]
