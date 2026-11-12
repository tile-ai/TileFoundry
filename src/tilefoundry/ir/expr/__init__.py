"""The IR foundation: expressions, operations, dimensions and the descriptors every layer shares.

Nothing here imports a type, a layout or a structure node; the layers above
extend it. See [core-ir](docs/spec/core-ir.md).
"""

from __future__ import annotations

from . import op_registry
from .communication import CommunicationOp
from .dim import (
    DIM_OP_TYPES,
    DimAdd,
    DimConst,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
    dim_variables,
    exact_quotient,
    is_dim_op_call,
    is_literal_shape,
    shape_has_dim_var,
    static_dim_value,
)
from .dtype import BoolDType, DType, FloatDType, IntegerDType, dtype_members
from .errors import VerifyError
from .expr import Call, Constant, Expr, Tuple, Var, child_exprs
from .kinds import BinaryKind, ReduceKind, UnaryKind
from .metadata import (
    Annotated,
    BindingMetadata,
    IRMetadata,
    RangeMetadata,
    SourceSpanMetadata,
    binding_name,
    describe_value,
    diagnostic_location,
    get_metadata,
    remove_metadata,
    source_metadata,
    value_label,
    value_labels,
)
from .op import (
    Op,
    OpCapability,
    ParameterInfo,
    capabilities_of,
    parameters_of,
    signature_of,
    supported_op_capabilities,
    target_capabilities,
)
from .op_registry import iter_schemas, schema_of
from .op_schema import OpSchema
from .param_def import (
    MISSING,
    MemoryEffect,
    ParamDef,
    ParamKind,
    collect_param_defs,
    variadic_item_annotation,
)
from .register import get_registered_name, register_op, register_type
from .storage import TARGET_MEMORY_OWNER, StorageKind, StorageOwners, resolve_storage
from .visitor import Functor, Visitor
