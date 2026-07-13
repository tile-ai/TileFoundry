"""Pointwise surface kinds: the ``log`` unary and the ``minimum`` / ``maximum``
binary aliases, exercised by composition oracles.

Resolution is pinned explicitly: ``exp`` stays the standalone first-class
``Exp`` op, ``log`` is ``Unary(LOG)``, and ``minimum`` / ``maximum`` are surface
aliases of the existing ``Binary`` MIN / MAX kinds.
"""
from __future__ import annotations

import torch

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.exp import Exp
from tilefoundry.ir.hir.math.unary import Unary

_DEV = "cpu"


# ── AC-2-1: sqrt(softplus(x)) via exp / log / rsqrt ──────────────────────────


@func
def _sqrt_softplus(x: Tensor[(4, 256), "f32"]) -> Tensor[(4, 256), "f32"]:
    # softplus(x) = log(1 + exp(x)); sqrt(y) = y * rsqrt(y)
    sp = log(add(exp(x), 1.0))  # noqa: F405
    return mul(sp, rsqrt(sp))  # noqa: F405


def test_sqrt_softplus_composition_matches_torch() -> None:
    """AC-2-1: ``sqrt(softplus(x))`` built from ``log`` / ``exp`` / ``rsqrt``
    matches torch on ``[4, 256] f32``."""
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    out = evaluate(_sqrt_softplus, x, device=_DEV)
    ref = torch.sqrt(torch.nn.functional.softplus(x))
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)


@func
def _exp_only(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return exp(x)  # noqa: F405


@func
def _log_only(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return log(x)  # noqa: F405


def test_exp_resolves_to_exp_op() -> None:
    """``exp`` stays the standalone first-class ``Exp`` op (not a Unary kind)."""
    body = _exp_only.body
    assert isinstance(body, Call) and isinstance(body.target, Exp)


def test_log_resolves_to_unary_log() -> None:
    """``log`` resolves to the generic ``Unary`` op with ``UnaryKind.LOG``."""
    body = _log_only.body
    assert isinstance(body, Call) and isinstance(body.target, Unary)
    assert body.target.kind is UnaryKind.LOG


# ── AC-2-2: asymmetric clamp via minimum / maximum ───────────────────────────


@func
def _min_clamp(g: Tensor[(4, 256), "f32"]) -> Tensor[(4, 256), "f32"]:
    return minimum(g, 10.0)  # noqa: F405


@func
def _asym_clamp(u: Tensor[(4, 256), "f32"]) -> Tensor[(4, 256), "f32"]:
    return maximum(minimum(u, 10.0), -10.0)  # noqa: F405


def test_min_clamp_matches_torch() -> None:
    """AC-2-2 (upper): ``minimum(g, 10)`` == ``torch.clamp(g, max=10)``."""
    torch.manual_seed(0)
    g = torch.randn(4, 256) * 20.0
    out = evaluate(_min_clamp, g, device=_DEV)
    torch.testing.assert_close(out.float(), torch.clamp(g, max=10.0), atol=1e-6, rtol=1e-6)


def test_asym_clamp_matches_torch() -> None:
    """AC-2-2 (asymmetric): ``maximum(minimum(u, 10), -10)`` ==
    ``torch.clamp(u, -10, 10)``."""
    torch.manual_seed(1)
    u = torch.randn(4, 256) * 20.0
    out = evaluate(_asym_clamp, u, device=_DEV)
    torch.testing.assert_close(
        out.float(), torch.clamp(u, min=-10.0, max=10.0), atol=1e-6, rtol=1e-6
    )


def test_minimum_maximum_resolve_to_binary_min_max() -> None:
    """``minimum`` / ``maximum`` are surface aliases of the existing ``Binary``
    MIN / MAX kinds."""
    lo = _min_clamp.body
    assert isinstance(lo, Call) and isinstance(lo.target, Binary)
    assert lo.target.kind is BinaryKind.MIN

    hi = _asym_clamp.body
    assert isinstance(hi, Call) and isinstance(hi.target, Binary)
    assert hi.target.kind is BinaryKind.MAX
    inner = hi.args[0]
    assert isinstance(inner, Call) and isinstance(inner.target, Binary)
    assert inner.target.kind is BinaryKind.MIN
