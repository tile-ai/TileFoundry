"""Op registry contract — DSL surface coverage."""
from __future__ import annotations

from tilefoundry.ir.core.op_registry import get_op_by_name, get_stmt_by_name

# Lock list of HIR / TIR DSL names — the registry must surface every
# one. Protects against an op accidentally losing its DSL surface
# during cleanup.
_HIR_REAL_OP_NAMES = frozenset({
    "argmax", "cast", "concat", "conv2d", "gather", "layer_norm", "local",
    "matmul", "quant", "rank", "relu", "reshape", "reshard", "rms_norm",
    "rope", "shape_of", "sigmoid", "slice", "softmax", "split", "stack",
    "tanh", "topk", "transpose",
})

_TIR_NAMES = frozenset({
    "copy", "fill", "alloc_tensor", "memory_span", "ptr_of",
    "tensor_view", "mma", "relu", "rms_norm", "reduce",
})


def test_dsl_surface_coverage_lock() -> None:
    """All HIR + TIR DSL names resolve."""
    for name in _HIR_REAL_OP_NAMES:
        assert get_op_by_name(name) is not None, f"HIR real Op {name!r} missing"
    for name in _TIR_NAMES:
        assert get_stmt_by_name(name) is not None, f"TIR {name!r} missing"
