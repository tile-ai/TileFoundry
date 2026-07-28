"""Sigmoid's Partial(R) commutation."""
from __future__ import annotations

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.sigmoid import Sigmoid
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_M = make_mesh((4,))


def test_sigmoid_rejects_partial_sum_input():
    """sigmoid is monotone increasing: it commutes with max/min, not sum."""
    run_typeinfer_case(
        TypeInferCase(
            "partial_sum_errors",
            Sigmoid(),
            (make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),)),),
            ExpectedError(match="Sigmoid"),
        )
    )
