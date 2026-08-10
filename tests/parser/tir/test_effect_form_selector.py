"""Pin the trailing-underscore effect selector on the real ``@prim_func`` path.

A bare ``copy_(...)`` unresolved through the closure strips its suffix and
resolves ``Copy`` from the T dialect, producing the same ``Evaluate`` statement
as ``T.copy(...)``. See [parser §1.3](docs/spec/parser.md#13-op-call) and
[parser §4.6](docs/spec/parser.md#46-per-dialect-strict-resolution).
"""

from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import Tensor
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.target import CpuTarget


def test_trailing_underscore_selects_effect_form_on_prim_func() -> None:
    @prim_func(target=CpuTarget())
    def dev(a: Tensor[(128,), "f32"], b: Tensor[(128,), "f32"]):
        copy_(a, b)  # noqa: F821 — resolved via dispatch.resolve_callable, not closure

    assert isinstance(dev.body, Sequential)
    (stmt,) = dev.body.body
    assert isinstance(stmt, Evaluate)
    assert isinstance(stmt.callable, Copy)
    assert stmt.args[0].name == "a"
    assert stmt.args[1].name == "b"
