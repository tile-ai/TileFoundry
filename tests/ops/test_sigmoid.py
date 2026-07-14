"""Sigmoid evaluator value oracle."""
from __future__ import annotations

import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tilefoundry.ir.hir.nn.sigmoid import Sigmoid


def test_sigmoid_evaluate():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    run_eval_case(EvalCase("sigmoid", Sigmoid(), (x,), torch.sigmoid(x), atol=1e-6))
