"""Rsqrt evaluator value oracle."""
from __future__ import annotations

import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.math.unary import Unary


def test_rsqrt_evaluate():
    torch.manual_seed(0)
    x = torch.rand(4) + 0.5
    run_eval_case(
        EvalCase("rsqrt", Unary(kind=UnaryKind.RSQRT), (x,), torch.rsqrt(x), atol=1e-6)
    )
