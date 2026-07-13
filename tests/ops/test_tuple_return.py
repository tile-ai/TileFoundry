"""Literal tuple return in ``@func``: the evaluator returns multiple values and
a caller destructures a two-output helper."""
from __future__ import annotations

import torch

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings for @func bodies
from tilefoundry.evaluator import evaluate

_DEV = "cpu"


@func
def _two_out(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]):
    return add(a, b), mul(a, b)  # noqa: F405


def test_tuple_return_evaluates_two_values() -> None:
    torch.manual_seed(0)
    a, b = torch.randn(4), torch.randn(4)
    out = evaluate(_two_out, a, b, device=_DEV)
    assert isinstance(out, tuple) and len(out) == 2
    torch.testing.assert_close(out[0], a + b)
    torch.testing.assert_close(out[1], a * b)


@func
def _caller(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    s, p = _two_out(a, b)
    return add(s, p)  # noqa: F405


def test_caller_destructures_two_output_helper() -> None:
    torch.manual_seed(1)
    a, b = torch.randn(4), torch.randn(4)
    out = evaluate(_caller, a, b, device=_DEV)
    torch.testing.assert_close(out, (a + b) + (a * b))
