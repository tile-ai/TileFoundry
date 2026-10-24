"""Softplus evaluator value oracle + Partial(R) commutation typeinfer.

Two corpus functions apply ``softplus``, and the one that distinguishes it -- the
KDA forget gate -- sits behind a blocked Reference, so that comparison is not
running. The value oracle stays here until it is.
"""

from __future__ import annotations

import torch

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir.hir.math.softplus import Softplus
from tilefoundry.ir.types import Layout, Mesh, Topology, make_shard_tensor_type
from tilefoundry.ir.types.shard_layout import Partial

_OP = Softplus()
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))


def _make_softplus_evaluate():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    return OpCase("softplus", Softplus(), (x,), value=torch.nn.functional.softplus(x), atol=1e-06)


CASES = (
    OpCase("partial_sum_errors", _OP, (_PSUM,), error=ExpectedError(match="Softplus")),
    _make_softplus_evaluate(),
)
test_op_case = op_case_test(CASES)
