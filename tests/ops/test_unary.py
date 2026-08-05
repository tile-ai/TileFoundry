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

from tests.evaluator.eval_utils import EvalCase, run_eval_case
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


#: The value oracles the corpus does not provide: no model calls ``abs``, and f16 is
#: a precision it never builds (bf16 sits beside f16 so a divergence between the two
#: low-precision paths is visible here). ``ROUND`` is torch's banker's rounding,
#: ties to even rather than half-away-from-zero, so it is asked on exact `.5` ties,
#: which a random input will not reliably hit: -2.5/-1.5 -> -2, -0.5/0.5 -> 0,
#: 1.5/2.5 -> 2.
ORACLES = [
    pytest.param("abs", _ABS, lambda: torch.randn(4), torch.abs, id="abs_f32"),
    pytest.param(
        "round_ties_to_even",
        Unary(kind=UnaryKind.ROUND),
        lambda: torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
        torch.round,
        id="round_ties_to_even",
    ),
    pytest.param(
        "exp", _EXP, lambda: torch.randn(4, dtype=torch.float16), torch.exp, id="exp_f16"
    ),
    pytest.param(
        "exp",
        _EXP,
        lambda: torch.randn(4, dtype=torch.bfloat16),
        torch.exp,
        id="exp_bf16",
    ),
]


@pytest.mark.parametrize(("name", "op", "drawn", "oracle"), ORACLES)
def test_a_unary_evaluates_to_its_oracle(name, op, drawn, oracle):
    torch.manual_seed(0)
    x = drawn()

    run_eval_case(EvalCase(name, op, (x,), oracle(x)))
