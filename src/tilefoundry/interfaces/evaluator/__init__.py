"""Reference evaluation: per-Op rules and the interpreter that runs them."""

from . import mesh_coord
from .context import EvaluateContext
from .dim import resolve_dim
from .interpreter import Visitor, evaluate, run_bound
from .kernels import binary, gather
from .registry import register, registry
from .utils import common_placements
from .value import (
    EvalError,
    TensorValue,
    TupleValue,
    Value,
    from_torch_dtype,
    tensor_type_of,
    to_torch_dtype,
)
