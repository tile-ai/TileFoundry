"""full_like evaluator value oracle.

``full_like(x, value)`` allocates a tensor shaped/typed like ``x`` filled with a
constant scalar — the DSL's way to seed loop-carry initial values (e.g. ``-inf``
running max) without a shape literal, so a dynamic (``DimVar``) extent needs none.
"""
from __future__ import annotations

import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tilefoundry.ir.hir.tensor.full_like import FullLike


def test_full_like_evaluate():
    x = torch.randn(2, 3)
    run_eval_case(
        EvalCase("full_like", FullLike(value=2.5), (x,), torch.full_like(x, 2.5))
    )


def test_full_like_preserves_dtype():
    x = torch.randn(3).bfloat16()
    run_eval_case(
        EvalCase(
            "full_like_bf16",
            FullLike(value=1.0),
            (x,),
            torch.ones(3, dtype=torch.bfloat16),
        )
    )
