"""Evaluator core mechanics: function-call binding, SSA identity, loop carry and
scalar-constant materialization.

Op-level value oracles live in the model References, which run whole decoders
through this same walker; what a real model's shape cannot make visible is kept
here on small parsed programs -- two params that are structurally equal, and a
carry whose init comes from the IR rather than from a first iteration.
"""
from __future__ import annotations

import torch

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, TensorType

_DEV = "cpu"


@func
def _add_scalar(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return add(x, 2.0)


def test_python_scalar_constant_operand():
    x = torch.randn(4)
    assert torch.allclose(evaluate(_add_scalar, x, device=_DEV), x + 2.0)


@func
def _callee(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return add(a, b)


@func
def _caller(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return mul(_callee(a, b), b)


def test_function_call_binds_callee_params():
    a, b = torch.randn(4), torch.randn(4)
    assert torch.allclose(evaluate(_caller, a, b, device=_DEV), (a + b) * b)


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

    result = evaluate(function, torch.tensor([1.0]), torch.tensor([2.0]), device=_DEV)

    assert torch.equal(result, torch.tensor([1.0]))


@func
def _carry_sum(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    acc = a
    for i in tile(3):
        acc = add(acc, b)
    return acc


def test_single_carry_accumulator():
    """Carry init comes from the IR's init_args (the param ``a``), looped 3×."""
    a, b = torch.randn(4), torch.randn(4)
    assert torch.allclose(evaluate(_carry_sum, a, b, device=_DEV), a + 3 * b)


@func
def _carry_two(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    p = a
    q = b
    for i in tile(2):
        p = add(p, b)
        q = add(q, a)
    return add(p, q)


def test_multi_carry_accumulator():
    """Two carries projected through TupleGetItem (a TupleValue)."""
    a, b = torch.randn(4), torch.randn(4)
    out = evaluate(_carry_two, a, b, device=_DEV)
    assert torch.allclose(out, (a + 2 * b) + (b + 2 * a))
