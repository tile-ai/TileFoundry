"""Sigmoid's Partial(R) commutation."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir.hir.nn.sigmoid import Sigmoid
from tilefoundry.ir.types import Layout, Mesh, Topology, make_shard_tensor_type
from tilefoundry.ir.types.shard_layout import Partial

_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
CASES = (
    OpCase(
        "partial_sum_errors",
        Sigmoid(),
        (make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),)),),
        error=ExpectedError(match="Sigmoid"),
    ),
)
test_op_case = op_case_test(CASES)
