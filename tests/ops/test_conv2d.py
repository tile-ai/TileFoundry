"""Conv2D typeinfer + Partial(R) commutation.

Conv2D is linear in each operand for the other held fixed (weight
replication), the same family as MatMul: a pre-existing ``Partial(sum)`` on
``input`` or ``weight`` propagates; ``max``/``min`` do not (convolution does
not preserve order).
"""
from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import ExpectedError, TypeInferCase, mesh, run_typeinfer_case, ten
from tilefoundry.ir.hir.nn.conv2d import Conv2D
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import Partial, ShardLayout

_OP = Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
_M = mesh((4,))
_F = DType.f32


def _sharded_nchw(shape, attrs) -> TensorType:
    return TensorType(
        shape=shape,
        dtype=_F,
        layout=ShardLayout(layout=Layout(shape=shape, strides=None), attrs=attrs, mesh=_M),
        storage="gmem",
    )


_X = ten((1, 4, 8, 8), _F)
_W = ten((4, 4, 3, 3), _F)
_BIAS = ten((4,), _F)

_X_PSUM = _sharded_nchw((1, 4, 8, 8), (Partial("sum"),))
_X_PMAX = _sharded_nchw((1, 4, 8, 8), (Partial("max"),))
_W_PSUM = _sharded_nchw((4, 4, 3, 3), (Partial("sum"),))
_W_PMAX = _sharded_nchw((4, 4, 3, 3), (Partial("max"),))

CASES = [
    TypeInferCase(
        "partial_sum_input_passes", _OP, (_X_PSUM, _W, _BIAS),
        ten((1, 4, 6, 6), _F, layout=_X_PSUM.layout),
    ),
    TypeInferCase(
        "partial_sum_weight_passes", _OP, (_X, _W_PSUM, _BIAS),
        ten((1, 4, 6, 6), _F, layout=_X.layout),
    ),
    TypeInferCase(
        "partial_max_input_errors", _OP, (_X_PMAX, _W, _BIAS),
        ExpectedError(match="Conv2D"),
    ),
    TypeInferCase(
        "partial_max_weight_errors", _OP, (_X, _W_PMAX, _BIAS),
        ExpectedError(match="Conv2D"),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_conv2d_typeinfer_partial(case):
    run_typeinfer_case(case)
