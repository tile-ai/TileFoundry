"""ReLU typeinfer + Partial(R) commutation."""

from __future__ import annotations

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir.hir.nn.relu import ReLU
from tilefoundry.ir.types import Layout, Mesh, Topology, make_shard_tensor_type
from tilefoundry.ir.types.shard_layout import Partial

_OP = ReLU()
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))
CASES = (
    OpCase("partial_max_passes", _OP, (_PMAX,), type=_PMAX),
    OpCase("partial_sum_errors", _OP, (_PSUM,), error=ExpectedError(match="ReLU")),
)
test_op_case = op_case_test(CASES)
