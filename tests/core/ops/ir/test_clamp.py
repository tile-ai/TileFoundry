"""Clamp typeinfer + Partial(R) commutation."""

from __future__ import annotations

import torch

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir.hir.math.clamp import Clamp
from tilefoundry.ir.types import Layout, Mesh, Topology, make_shard_tensor_type
from tilefoundry.ir.types.shard_layout import Partial

_OP = Clamp(min_val=-1.0, max_val=1.0)
_M = Mesh((Topology("gpu", 4),), Layout((4,), (1,)), ("g",))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))


def _make_clamp_scalar_bounds_match_torch_reference() -> OpCase:
    x = torch.linspace(-2.0, 2.0, 32).reshape(4, 8)
    return OpCase("scalar_bounds", _OP, (x,), value=torch.clamp(x, -1.0, 1.0))


CASES = (
    _make_clamp_scalar_bounds_match_torch_reference(),
    OpCase("partial_max_passes", _OP, (_PMAX,), type=_PMAX),
    OpCase("partial_sum_errors", _OP, (_PSUM,), error=ExpectedError(match="Clamp")),
)
test_op_case = op_case_test(CASES)
