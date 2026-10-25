"""Core scalar, call, and loop evaluation semantics."""

from __future__ import annotations

from pathlib import Path

import torch

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Var
from tilefoundry.ir.hir.function import Function

CORE = Path(__file__).parents[1] / "fixtures" / "evaluator" / "core_semantics.py"


def test_python_scalar_constant_operand():
    add_scalar = load_authored_ir(f"{CORE}:AddScalar").entry_function()
    x = torch.randn(4)
    assert torch.allclose(evaluate(add_scalar, x), x + 2.0)


def test_function_call_binds_callee_params():
    caller = load_authored_ir(f"{CORE}:CallerCallee").entry_function()
    a, b = torch.randn(4), torch.randn(4)
    assert torch.allclose(evaluate(caller, a, b), (a + b) * b)


def test_structurally_equal_params_keep_distinct_ssa_bindings():
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


def test_single_carry_accumulator():
    """Carry init comes from the IR's args (the param ``a``), looped 3×."""
    carry_sum = load_authored_ir(f"{CORE}:CarrySum").entry_function()
    a, b = torch.randn(4), torch.randn(4)
    assert torch.allclose(evaluate(carry_sum, a, b), a + 3 * b)


def test_multi_carry_accumulator():
    """Two carries projected through TupleGetItem (a TupleValue)."""
    carry_two = load_authored_ir(f"{CORE}:CarryTwo").entry_function()
    a, b = torch.randn(4), torch.randn(4)
    out = evaluate(carry_two, a, b)
    assert torch.allclose(out, (a + 2 * b) + (b + 2 * a))
