"""Type inference: rules registered per Op, and the visitor that applies them."""

from . import dim, mesh_coord
from .anchor_storage import resolve_anchor_storage
from .context import TypeInferContext, TypeInferResults
from .reachability import check_input_reachability
from .registry import register, registry
from .shard_propagate import (
    combine_operand_attrs,
    derive_output_shard_layout,
    induction_type,
    operand_types,
)
from .visitor import TypeInferVisitor, infer_function, inference_type
