"""Silu evaluator value oracle + Partial(R) commutation typeinfer."""

from __future__ import annotations

import torch

from tests.core.ops.ir.op_case import ExpectedError, OpCase, check_op_case, op_case_test
from tilefoundry.ir import types, utils
from tilefoundry.ir.hir import nn

_OP = nn.Silu()
_M = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)), ("g",))
_PSUM = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("sum"),))
_PMAX = utils.make_shard_tensor_type((16, 8), mesh=_M, attrs=(types.Partial("max"),))


def test_silu_evaluate():
    """Within tolerance at f32, and bit for bit at bf16."""
    torch.manual_seed(0)
    x = torch.randn(4)
    check_op_case(OpCase("silu", nn.Silu(), (x,), value=torch.nn.functional.silu(x), atol=1e-06))
    x_bf16 = (torch.randn(10000) * 0.003).to(torch.bfloat16)
    check_op_case(
        OpCase(
            "silu_bf16", nn.Silu(), (x_bf16,), value=torch.nn.functional.silu(x_bf16), atol=0, rtol=0
        )
    )


def test_silu_differs_from_decomposed():
    """The fused and decomposed forms are not bit-identical at bf16."""
    torch.manual_seed(0)
    x = (torch.randn(10000) * 0.003).to(torch.bfloat16)
    fused = torch.nn.functional.silu(x)
    decomposed = x * torch.sigmoid(x)
    assert not torch.equal(fused, decomposed)
    assert (fused.float() - decomposed.float()).abs().max().item() > 0


CASES = (
    OpCase("partial_max_errors", _OP, (_PMAX,), error=ExpectedError(match="Silu")),
    OpCase("partial_sum_errors", _OP, (_PSUM,), error=ExpectedError(match="Silu")),
)
test_op_case = op_case_test(CASES)
