"""Clamp typeinfer + Partial(R) commutation."""

from __future__ import annotations

import torch

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.ir import types, utils
from tilefoundry.ir.hir import math

_OP = math.Clamp(min_val=-1.0, max_val=1.0)
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
_PSUM = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("sum"),))
_PMAX = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("max"),))


def _make_clamp_scalar_bounds_match_torch_reference() -> OpCase:
    x = torch.linspace(-2.0, 2.0, 32).reshape(4, 8)
    return OpCase("scalar_bounds", _OP, (x,), value=torch.clamp(x, -1.0, 1.0))


CASES = (
    _make_clamp_scalar_bounds_match_torch_reference(),
    OpCase("partial_max_passes", _OP, (_PMAX,), type=_PMAX),
    OpCase("partial_sum_errors", _OP, (_PSUM,), error=ExpectedError(match="Clamp")),
)
test_op_case = op_case_test(CASES)
