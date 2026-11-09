"""HIR SoftMax's Partial-input rejection."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir import types, utils
from tilefoundry.ir.hir import nn


def _make_softmax_typeinfer_partial_input_errors():
    m = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
    return OpCase(
        "partial_sum_errors",
        nn.SoftMax(axis=-1),
        (utils.make_shard_tensor_type((2, 8), mesh=m, attrs=(types.Partial("sum"),)),),
        error=ExpectedError(match="SoftMax"),
    )


CASES = (_make_softmax_typeinfer_partial_input_errors(),)
test_op_case = op_case_test(CASES)
