"""LayerNorm typeinfer + Partial(R) commutation."""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, sharded, ten
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = LayerNorm(axis=-1, eps=1e-5)
_F = DType.f32
_X = ten((4, 8), _F)
_W = ten((8,), _F)
_B = ten((8,), _F)

CASES = [
    TypeInferCase("passthrough", _OP, (_X, _W, _B), _X),
    # layer_norm normalizes across an axis (non-monotonic); no reduction
    # commutes.
    TypeInferCase(
        "partial_sum_errors",
        _OP,
        (sharded((4, 8), (Partial("sum"),), mesh((4,))), _W, _B),
        ExpectedError(match="LayerNorm"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_layer_norm_typeinfer(case):
    run_typeinfer_case(case)
