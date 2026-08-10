"""HIR SoftMax's Partial-input rejection."""

from __future__ import annotations

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.softmax import SoftMax
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial


def test_softmax_typeinfer_partial_input_errors():

    m = make_mesh((4,))
    run_typeinfer_case(
        TypeInferCase(
            "partial_sum_errors",
            SoftMax(axis=-1),
            (make_shard_tensor_type((2, 8), mesh=m, attrs=(Partial("sum"),)),),
            ExpectedError(match="SoftMax"),
        )
    )
