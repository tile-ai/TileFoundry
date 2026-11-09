"""RepeatInterleave's fail-closed boundaries.

The growing axis invalidates the input layout, so a genuinely-sharded input
fails closed rather than emit a stale layout, and an out-of-range axis is named
rather than normalized.
"""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import tensor

_F = expr.DType.f32
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
CASES = (
    OpCase(
        "sharded_fails_closed",
        tensor.RepeatInterleave(repeats=2, axis=1),
        (utils.make_shard_tensor_type((4, 8), mesh=_M, attrs=(types.Split(0),)),),
        error=ExpectedError(match="cannot express a sharded layout"),
    ),
    OpCase(
        "axis_out_of_range",
        tensor.RepeatInterleave(repeats=2, axis=5),
        (types.TensorType.from_shape((4,), _F),),
        error=ExpectedError(match="out of range"),
    ),
)
test_op_case = op_case_test(CASES)
