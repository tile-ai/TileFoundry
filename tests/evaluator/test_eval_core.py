"""Evaluator core mechanics.

Evaluator core mechanics: function-call binding, SSA identity, loop carry and
scalar-constant materialization.

Op-level value oracles live in the model References, which run whole decoders
through this same walker; what a real model's shape cannot make visible is kept
here on small parsed programs -- two params that are structurally equal, and a
carry whose init comes from the IR rather than from a first iteration.
"""

from __future__ import annotations

import torch

from tests.fixtures.shapes.scaled_modules import (
    EVALUATOR_N as _N_EVAL,
)
from tests.fixtures.shapes.scaled_modules import (
    DynamicScaledChild,
    FusedScaledParent,
    PairedScaledParent,
    ScaledChild,
)
from tilefoundry import func, module
from tilefoundry.dsl import DimVarRangePat, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.target import CudaTarget

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
    for i in range(3):
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
    for i in range(2):
        p = add(p, b)
        q = add(q, a)
    return add(p, q)


def test_multi_carry_accumulator():
    """Two carries projected through TupleGetItem (a TupleValue)."""
    a, b = torch.randn(4), torch.randn(4)
    out = evaluate(_carry_two, a, b, device=_DEV)
    assert torch.allclose(out, (a + 2 * b) + (b + 2 * a))


class _Weights:
    """A resource whose subtree is the dotted prefix, like a checkpoint's."""

    def __init__(self, values: dict) -> None:
        self.values = values

    def load(self, name: str):
        return self.values[name]

    def subtree(self, name: str) -> "_Weights":
        prefix = f"{name}."
        return _Weights(
            {k[len(prefix):]: v for k, v in self.values.items() if k.startswith(prefix)}
        )


def test_a_child_module_call_runs_against_that_child_reading() -> None:
    x = torch.arange(4, dtype=torch.float32)
    w = torch.tensor([2.0, 3.0, 4.0, 5.0])
    reading = FusedScaledParent.load(_Weights({"scaled.w": w}))

    assert torch.equal(reading.fused(x), x * w)


def test_one_module_read_twice_yields_two_independent_readings() -> None:
    x = torch.ones(4)
    first = FusedScaledParent.load(_Weights({"scaled.w": torch.full((4,), 2.0)}))
    second = FusedScaledParent.load(_Weights({"scaled.w": torch.full((4,), 5.0)}))

    assert torch.equal(second.fused(x), torch.full((4,), 5.0))
    assert torch.equal(first.fused(x), torch.full((4,), 2.0))


def test_two_bindings_of_one_child_read_their_own_constants() -> None:
    x = torch.ones(4)
    reading = PairedScaledParent.load(
        _Weights({"left.w": torch.full((4,), 2.0), "right.w": torch.full((4,), 7.0)})
    )

    assert torch.equal(reading.both(x), torch.full((4,), 9.0))


def test_a_child_call_inside_a_loop_keeps_its_reading_on_every_trip() -> None:
    @module(entry="looped", target=CudaTarget("nvidia.h200_sxm"))
    class _Looped:
        scaled = ScaledChild

        @func
        def looped(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            acc = x
            for _ in range(3):
                acc = scaled(acc)  # noqa: F821
            return acc

    x = torch.ones(4)
    w = torch.full((4,), 2.0)
    reading = _Looped.load(_Weights({"scaled.w": w}))

    assert torch.equal(reading.looped(x), x * w * w * w)


def test_a_variant_body_reaches_its_child_the_same_way() -> None:
    @module(entry="dispatch", target=CudaTarget("nvidia.h200_sxm"))
    class _Dispatch:
        scaled = DynamicScaledChild

        @func
        def dispatch(x: Tensor[(_N_EVAL,), "f32"]) -> Tensor[(_N_EVAL,), "f32"]:
            pass

        @dispatch.specialize(DimVarRangePat("N_eval", 1, 8))
        def dynamic_variant(x: Tensor[(_N_EVAL,), "f32"]) -> Tensor[(_N_EVAL,), "f32"]:
            return scaled(x)  # noqa: F821

    (child,) = _Dispatch.modules
    (variant,) = _Dispatch.entry_function().variants
    assert variant.body.target is child.entry_function()
    assert len(variant.body.args) == 1
    assert [p.is_const for p in variant.body.target.params] == [False, True]
