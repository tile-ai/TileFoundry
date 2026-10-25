"""Core scalar, call, and loop evaluation semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Var
from tilefoundry.ir.hir.function import Function

CORE = Path(__file__).parents[1] / "fixtures" / "evaluator" / "core_semantics.py"


class CoreEvalCase(Case):
    """A core-semantics program evaluated against its torch reference."""


@case
class PythonScalarConstantOperand(CoreEvalCase):
    path: Path = CORE
    name: str | None = "AddScalar"

    def guard(self, state, problems):
        add_scalar = state.entry_function()
        x = torch.randn(4)
        assert torch.allclose(evaluate(add_scalar, x), x + 2.0)


@case
class CallBindsCalleeParams(CoreEvalCase):
    path: Path = CORE
    name: str | None = "CallerCallee"

    def guard(self, state, problems):
        caller = state.entry_function()
        a, b = torch.randn(4), torch.randn(4)
        assert torch.allclose(evaluate(caller, a, b), (a + b) * b)


def test_equal_params_keep_distinct_bindings():
    tensor_type = TensorType((1,), DType.f32, layout=None, storage="gmem")
    first = Var(name="same", type=tensor_type)
    second = Var(name="same", type=tensor_type)
    function = Function.build(
        name="same_named_params",
        params=(first, second),
        body=first,
        return_type=tensor_type,
    )

    result = evaluate(function, torch.tensor([1.0]), torch.tensor([2.0]))

    assert torch.equal(result, torch.tensor([1.0]))


@case
class SingleCarryAccumulator(CoreEvalCase):
    """Carry init comes from the IR's args (the param ``a``), looped 3×."""

    path: Path = CORE
    name: str | None = "CarrySum"

    def guard(self, state, problems):
        carry_sum = state.entry_function()
        a, b = torch.randn(4), torch.randn(4)
        assert torch.allclose(evaluate(carry_sum, a, b), a + 3 * b)


@case
class MultiCarryAccumulator(CoreEvalCase):
    """Two carries projected through TupleGetItem (a TupleValue)."""

    path: Path = CORE
    name: str | None = "CarryTwo"

    def guard(self, state, problems):
        carry_two = state.entry_function()
        a, b = torch.randn(4), torch.randn(4)
        out = evaluate(carry_two, a, b)
        assert torch.allclose(out, (a + 2 * b) + (b + 2 * a))


CASES = special(CoreEvalCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_core_eval_case(case: CoreEvalCase) -> None:
    check_case(case)
