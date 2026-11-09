"""ReLU typeinfer + Partial(R) commutation."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir import types, utils
from tilefoundry.ir.hir import nn

_OP = nn.ReLU()
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
_PSUM = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("sum"),))
_PMAX = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("max"),))
CASES = (
    OpCase("partial_max_passes", _OP, (_PMAX,), type=_PMAX),
    OpCase("partial_sum_errors", _OP, (_PSUM,), error=ExpectedError(match="ReLU")),
)
test_op_case = op_case_test(CASES)
