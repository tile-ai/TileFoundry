"""Silu evaluator value oracle + Partial(R) commutation typeinfer."""
from __future__ import annotations

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.hir.nn.silu import Silu
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_OP = Silu()
_M = make_mesh((4,))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))

PARTIAL_CASES = [
    # Both reductions are rejected; Sigmoid / Softplus pass Partial(max).
    TypeInferCase("partial_max_errors", _OP, (_PMAX,), ExpectedError(match="Silu")),
    TypeInferCase("partial_sum_errors", _OP, (_PSUM,), ExpectedError(match="Silu")),
]


@pytest.mark.parametrize("case", PARTIAL_CASES, ids=lambda c: c.name)
def test_silu_typeinfer_partial(case):
    run_typeinfer_case(case)


def test_silu_evaluate():
    """Within tolerance at f32, and bit for bit at bf16."""
    torch.manual_seed(0)
    x = torch.randn(4)
    run_eval_case(EvalCase("silu", Silu(), (x,), torch.nn.functional.silu(x), atol=1e-6))

    x_bf16 = (torch.randn(10_000) * 0.003).to(torch.bfloat16)
    run_eval_case(
        EvalCase(
            "silu_bf16", Silu(), (x_bf16,),
            torch.nn.functional.silu(x_bf16), atol=0, rtol=0,
        )
    )


def test_silu_differs_from_decomposed_form_at_bf16():
    """The fused and decomposed forms are not bit-identical at bf16."""
    torch.manual_seed(0)
    x = (torch.randn(10_000) * 0.003).to(torch.bfloat16)
    fused = torch.nn.functional.silu(x)
    decomposed = x * torch.sigmoid(x)
    assert not torch.equal(fused, decomposed)
    assert (fused.float() - decomposed.float()).abs().max().item() > 0
