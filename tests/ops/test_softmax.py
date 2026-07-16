"""HIR SoftMax value oracle: f32-accumulated softmax along the named axis."""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.softmax import SoftMax
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial


@pytest.mark.parametrize(
    "shape,axis",
    [((2, 4, 8), -1), ((3, 6), 1)],
    ids=["softmax_last_axis", "softmax_mid_axis"],
)
def test_softmax_evaluate(shape, axis):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    run_eval_case(EvalCase("", SoftMax(axis=axis), (x,), torch.softmax(x, dim=axis)))


def test_softmax_typeinfer_partial_input_errors():
    # softmax normalizes across an axis (non-monotonic); no reduction commutes.
    m = make_mesh((4,))
    run_typeinfer_case(
        TypeInferCase(
            "partial_sum_errors",
            SoftMax(axis=-1),
            (make_shard_tensor_type((2, 8), mesh=m, attrs=(Partial("sum"),)),),
            ExpectedError(match="SoftMax"),
        )
    )
