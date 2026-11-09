"""Sigmoid's Partial(R) commutation."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir import types, utils
from tilefoundry.ir.hir import nn

_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
CASES = (
    OpCase(
        "partial_sum_errors",
        nn.Sigmoid(),
        (utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("sum"),)),),
        error=ExpectedError(match="Sigmoid"),
    ),
)
test_op_case = op_case_test(CASES)
