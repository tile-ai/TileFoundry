"""RepeatInterleave's fail-closed boundaries.

The growing axis invalidates the input layout, so a genuinely-sharded input
fails closed rather than emit a stale layout, and an out-of-range axis is named
rather than normalized.
"""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir.hir.tensor.repeat_interleave import RepeatInterleave
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    Topology,
    make_shard_tensor_type,
    make_tensor_type,
)
from tilefoundry.ir.types.shard_layout import Split

_F = DType.f32
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
CASES = (
    OpCase(
        "sharded_fails_closed",
        RepeatInterleave(repeats=2, axis=1),
        (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Split(0),)),),
        error=ExpectedError(match="cannot express a sharded layout"),
    ),
    OpCase(
        "axis_out_of_range",
        RepeatInterleave(repeats=2, axis=5),
        (make_tensor_type((4,), _F),),
        error=ExpectedError(match="out of range"),
    ),
)
test_op_case = op_case_test(CASES)
