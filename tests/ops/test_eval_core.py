"""Evaluator core mechanics: node walking, memoization, function-call and
loop-carry semantics, constant materialization, and the layout-view helpers.

Op-level value oracles live in the per-op ``tests/ops/test_<op>.py`` files;
this file exercises the walker itself on small parsed programs.
"""
from __future__ import annotations

import torch

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.evaluator import (
    EvalError,
    TensorValue,
    as_layout_view,
    evaluate,
    from_layout_view,
)
from tilefoundry.evaluator.registry import eval_registry
from tilefoundry.ir.core import Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types import DType, TensorType

_DEV = "cpu"


@func
def _add(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return add(a, b)


def test_single_binary_add_matches_torch():
    a, b = torch.randn(4), torch.randn(4)
    assert torch.allclose(evaluate(_add, a, b, device=_DEV), a + b)


@func
def _add_scalar(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return add(x, 2.0)


def test_python_scalar_constant_operand():
    x = torch.randn(4)
    assert torch.allclose(evaluate(_add_scalar, x, device=_DEV), x + 2.0)


@func
def _shared(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    c = add(a, b)
    return mul(c, c)


def test_shared_subexpr_evaluated_once(monkeypatch):
    """The reused ``add`` is evaluated exactly once (memoized): the Binary
    handler runs twice (one add + one mul), not three times."""
    original = eval_registry.lookup(Binary)
    calls = {"n": 0}

    def counting(ctx):
        calls["n"] += 1
        return original(ctx)

    monkeypatch.setitem(eval_registry._map, Binary, counting)
    a, b = torch.randn(4), torch.randn(4)
    out = evaluate(_shared, a, b, device=_DEV)
    assert torch.allclose(out, (a + b) * (a + b))
    assert calls["n"] == 2


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


@func
def _zeros_fn(x: Tensor[(2, 3), "f32"]) -> Tensor[(2, 3), "f32"]:
    return add(x, zeros((2, 3), "f32"))


def test_zeros():
    x = torch.randn(2, 3)
    assert torch.allclose(evaluate(_zeros_fn, x, device=_DEV), x)


def test_unregistered_op_raises_naming_op():
    @func
    def uses_relu(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return relu(x)

    try:
        evaluate(uses_relu, torch.randn(4), device=_DEV)
    except EvalError as e:
        assert "ReLU" in str(e)
    else:
        raise AssertionError("expected EvalError for unregistered op")


def test_layout_view_roundtrip():
    """No-layout values pass through; from_layout_view restores logical shape."""
    t = TensorType(shape=(2, 3), dtype=DType.f32, layout=None, storage="gmem")
    v = TensorValue(data=torch.randn(2, 3), type=t)
    assert torch.equal(as_layout_view(v), v.data)
    assert tuple(from_layout_view(v.data.reshape(6), t).shape) == (2, 3)
