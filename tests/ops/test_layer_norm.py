"""LayerNorm typeinfer + Partial(R) commutation."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = LayerNorm(axis=-1, eps=1e-5)
_F = DType.f32
_M = make_mesh((4,))
_X = make_tensor_type((4, 8), _F)
_W = make_tensor_type((8,), _F)
_B = make_tensor_type((8,), _F)
_W_PSUM = make_shard_tensor_type((8,), mesh=_M, attrs=(Partial("sum"),))
_B_PSUM = make_shard_tensor_type((8,), mesh=_M, attrs=(Partial("sum"),))

CASES = [
    TypeInferCase("passthrough", _OP, (_X, _W, _B), _X),
    # layer_norm normalizes across an axis (non-monotonic); no reduction
    # commutes.
    TypeInferCase(
        "partial_sum_errors",
        _OP,
        (make_shard_tensor_type((4, 8), mesh=_M, attrs=(Partial("sum"),)), _W, _B),
        ExpectedError(match="LayerNorm"),
    ),
    TypeInferCase(
        "partial_weight_errors",
        _OP,
        (_X, _W_PSUM, _B),
        ExpectedError(match="weight.*Partial.*mesh axis 0"),
    ),
    TypeInferCase(
        "partial_bias_errors",
        _OP,
        (_X, _W, _B_PSUM),
        ExpectedError(match="bias.*Partial.*mesh axis 0"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_layer_norm_typeinfer(case):
    run_typeinfer_case(case)
