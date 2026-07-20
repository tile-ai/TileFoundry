"""Trailing-underscore effect-form selector (parser.md §1.3/§4.6) on the real
``@prim_func`` path.

``dispatch.resolve_callable`` implements the ``foo_`` convention, but only
``tests/core/test_op_registry.py`` exercised it directly before this — the
actual parser never routed a callee through it. This locks the selector
against the real ``_TirBodyVisitor`` entry point: a bare ``copy_(...)`` call,
unresolved through the closure, strips the trailing underscore and resolves
`Copy` through the T-dialect registry directly, producing the same
``Evaluate``-wrapped Stmt as the namespaced ``T.copy(...)`` form.
"""
from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import Tensor
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.stmts import Evaluate, Sequential


def test_trailing_underscore_selects_effect_form_on_prim_func() -> None:
    @prim_func(target="cpu")
    def dev(a: Tensor[(128,), "f32"], b: Tensor[(128,), "f32"]):
        copy_(a, b)  # noqa: F821 — resolved via dispatch.resolve_callable, not closure

    assert isinstance(dev.body, Sequential)
    (stmt,) = dev.body.body
    assert isinstance(stmt, Evaluate)
    assert isinstance(stmt.callable, Copy)
    assert stmt.args[0].name == "a"
    assert stmt.args[1].name == "b"
