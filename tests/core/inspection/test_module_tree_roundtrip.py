"""Target-neutral tests and helpers."""

from __future__ import annotations

from tests.core._source import import_dsl
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script


@module(entry="caller")
class _Siblings:
    """A callee declared above its caller.

    A callee declared above its caller, which is what the printer emits for
    any multi-function Module.
    """

    @func
    def callee(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @func
    def caller(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return callee(x)


def test_a_sibling_call_survives_the_round_trip() -> None:
    """A body calling an earlier sibling is canonical output.

    A body calling an earlier sibling is canonical output, so importing it
    must resolve the callee to that sibling rather than read it as an op name.
    """
    imported = import_dsl(as_script(_Siblings))

    assert [fn.name for fn in imported.functions] == ["callee", "caller"]
    callee, caller = imported.functions
    assert caller.body.target is callee

    once = as_script(imported)
    assert as_script(import_dsl(once)) == once
