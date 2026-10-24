"""HIR SoftMax's Partial-input rejection."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir.hir.nn.softmax import SoftMax
from tilefoundry.ir.types import Layout, Mesh, Topology, make_shard_tensor_type
from tilefoundry.ir.types.shard_layout import Partial


def _make_softmax_typeinfer_partial_input_errors():
    m = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
    return OpCase(
        "partial_sum_errors",
        SoftMax(axis=-1),
        (make_shard_tensor_type((2, 8), mesh=m, attrs=(Partial("sum"),)),),
        error=ExpectedError(match="SoftMax"),
    )


CASES = (_make_softmax_typeinfer_partial_input_errors(),)
test_op_case = op_case_test(CASES)
