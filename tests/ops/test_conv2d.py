"""Conv2D typeinfer + Partial(R) commutation.

Conv2D is linear in each operand for the other held fixed (weight
replication), the same family as MatMul: a pre-existing ``Partial(sum)`` on
``input`` or ``weight`` propagates; ``max``/``min`` do not (convolution does
not preserve order).
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, run_typeinfer_case
from tilefoundry.ir.hir.nn.conv2d import Conv2D
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial, Split

_OP = Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
_M = make_mesh((4,))
_F = DType.f32

_X = make_tensor_type((1, 4, 8, 8), _F)
_W = make_tensor_type((4, 4, 3, 3), _F)
_BIAS = make_tensor_type((4,), _F)

_X_PSUM = make_shard_tensor_type((1, 4, 8, 8), mesh=_M, attrs=(Partial("sum"),), dtype=_F)
_X_PMAX = make_shard_tensor_type((1, 4, 8, 8), mesh=_M, attrs=(Partial("max"),), dtype=_F)
_W_PSUM = make_shard_tensor_type((4, 4, 3, 3), mesh=_M, attrs=(Partial("sum"),), dtype=_F)
_W_SPLIT = make_shard_tensor_type((4, 4, 3, 3), mesh=_M, attrs=(Split(0),), dtype=_F)
_BIAS_PSUM = make_shard_tensor_type((4,), mesh=_M, attrs=(Partial("sum"),), dtype=_F)

CASES = [
    TypeInferCase(
        "partial_sum_input_passes", _OP, (_X_PSUM, _W, _BIAS),
        make_tensor_type((1, 4, 6, 6), _F, layout=_X_PSUM.layout),
    ),
    TypeInferCase(
        "partial_sum_weight_is_rejected_as_secondary",
        _OP,
        (_X, _W_PSUM, _BIAS),
        ExpectedError(match="weight carries Partial.*mesh axis 0"),
    ),
    TypeInferCase(
        "partial_max_input_errors", _OP, (_X_PMAX, _W, _BIAS),
        ExpectedError(match="Conv2D"),
    ),
    TypeInferCase(
        "partial_input_with_split_weight_errors",
        _OP,
        (_X_PSUM, _W_SPLIT, _BIAS),
        ExpectedError(match="weight is not Broadcast/replicated on that axis"),
    ),
    TypeInferCase(
        "partial_bias_is_rejected_as_secondary",
        _OP,
        (_X, _W, _BIAS_PSUM),
        ExpectedError(match="bias carries Partial.*mesh axis 0"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_conv2d_typeinfer_partial(case):
    run_typeinfer_case(case)
