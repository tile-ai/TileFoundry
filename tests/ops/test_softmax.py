"""HIR SoftMax value oracle: f32-accumulated softmax along the named axis."""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tilefoundry.ir.hir.nn.softmax import SoftMax


@pytest.mark.parametrize(
    "shape,axis",
    [((2, 4, 8), -1), ((3, 6), 1)],
    ids=["softmax_last_axis", "softmax_mid_axis"],
)
def test_softmax_evaluate(shape, axis):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    run_eval_case(EvalCase("", SoftMax(axis=axis), (x,), torch.softmax(x, dim=axis)))
