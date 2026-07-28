"""Unary's Partial algebra by class, its low-precision dtype, and the two kinds
no corpus model uses.

Shape / layout / storage pass through the input, and the corpus decoders apply
``exp``, ``neg``, ``square``, ``log``, ``log2``, ``exp2``, ``rsqrt`` and ``ceil``
on real inputs, so the model References witness the plain values. What stays here
is what they cannot show:

- the Partial algebra, one case per *class* rather than per kind: sign-preserving
  linear (NEG) commutes with ``sum`` and not ``max``; monotone non-decreasing
  (EXP, and identically CEIL / ROUND / EXP2 / LOG2) commutes with ``max`` and not
  ``sum``; neither (ABS, RSQRT) commutes with either;
- a low-precision operand: inference is purely logical, so an ``fp8`` element
  type passes through like any other;
- ``ABS`` and ``ROUND``, which no corpus model calls, keep a value oracle -- and
  ``ROUND``'s is on exact ``.5`` ties, which is the whole question about it;
- ``f16``, which no corpus model uses.
"""
from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Partial

_NEG = Unary(kind=UnaryKind.NEG)
_EXP = Unary(kind=UnaryKind.EXP)
_ABS = Unary(kind=UnaryKind.ABS)
_RSQRT = Unary(kind=UnaryKind.RSQRT)
_M = make_mesh((4,))
_PSUM = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("sum"),))
_PMAX = make_shard_tensor_type((16, 8), mesh=_M, attrs=(Partial("max"),))


CASES = [
    # Low-precision dtypes are legal typeinfer operands: inference is purely
    # logical, so they pass through like any other element type.
    TypeInferCase(
        "low_precision_passthrough_fp8e4m3",
        _NEG,
        (make_tensor_type((4, 8), DType.fp8e4m3),),
        make_tensor_type((4, 8), DType.fp8e4m3),
    ),
    TypeInferCase("neg_partial_sum_passes", _NEG, (_PSUM,), _PSUM),
    TypeInferCase(
        "neg_partial_max_errors", _NEG, (_PMAX,), ExpectedError(match="carries Partial")
    ),
    # EXP stands for CEIL / ROUND / EXP2 / LOG2 as well: all monotone
    # non-decreasing, one class, one rule.
    TypeInferCase("exp_partial_max_passes", _EXP, (_PMAX,), _PMAX),
    TypeInferCase(
        "exp_partial_sum_errors", _EXP, (_PSUM,), ExpectedError(match="carries Partial")
    ),
    TypeInferCase(
        "abs_partial_sum_errors", _ABS, (_PSUM,), ExpectedError(match="carries Partial")
    ),
    TypeInferCase(
        "rsqrt_partial_max_errors", _RSQRT, (_PMAX,), ExpectedError(match="carries Partial")
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_unary_typeinfer(case):
    run_typeinfer_case(case)


def test_unary_evaluate_abs():
    """No corpus model calls ``abs``, so its value oracle stays here."""
    torch.manual_seed(0)
    x = torch.randn(4)
    run_eval_case(EvalCase("abs", _ABS, (x,), x.abs()))


def test_unary_evaluate_round_half_to_even():
    """``ROUND`` uses torch's banker's rounding (ties to even), not
    round-half-away-from-zero -- exercised on exact `.5` ties, which a random
    input will not reliably hit. Expected values: -2.5/-1.5 -> -2, -0.5/0.5 -> 0,
    1.5/2.5 -> 2 (each tie rounds to the nearest *even* integer)."""
    x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    run_eval_case(EvalCase("round_ties_to_even", Unary(kind=UnaryKind.ROUND), (x,), x.round()))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
def test_unary_evaluate_dtypes(dtype):
    """The corpus runs bf16 and f32; f16 it never builds. bf16 stays alongside it
    so a divergence between the two low-precision paths is visible here."""
    torch.manual_seed(0)
    x = torch.randn(4, dtype=dtype)
    run_eval_case(EvalCase("exp", _EXP, (x,), torch.exp(x)))
