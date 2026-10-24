"""Tanh typeinfer + Partial(R) commutation, and what it evaluates to."""

from __future__ import annotations

import torch

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir.hir.nn.tanh import Tanh
from tilefoundry.ir.types import Layout, Mesh, Topology, make_shard_tensor_type
from tilefoundry.ir.types.shard_layout import Partial

_OP = Tanh()
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))
_VALUES = torch.tensor(
    [-8.0, -1.0, -0.25, -0.03125, -0.0078125, 0.0, 0.0078125, 0.03125, 1.0, 8.0],
    dtype=torch.bfloat16,
)
CASES = (
    OpCase("partial_max_passes", _OP, (_PMAX,), type=_PMAX),
    OpCase("partial_sum_errors", _OP, (_PSUM,), error=ExpectedError(match="Tanh")),
    OpCase(
        "bf16_matches_torch_tanh", Tanh(), (_VALUES,), value=torch.tanh(_VALUES), atol=0, rtol=0
    ),
    OpCase(
        "f32_matches_torch_tanh",
        Tanh(),
        (_VALUES.float(),),
        value=torch.tanh(_VALUES.float()),
        atol=0,
        rtol=0,
    ),
)
test_op_case = op_case_test(CASES)
