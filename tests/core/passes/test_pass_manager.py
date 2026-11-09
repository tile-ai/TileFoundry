"""Coverage for tilefoundry.passes — Pass / PassManager orchestration.

Per-transform behaviour lives in the transform's own test file, and the default
``tilefoundry.build`` pipeline is exercised end to end by every test that
compiles and runs a kernel: a pipeline that stopped wiring itself up cannot
produce a kernel at all, so it needs no separate assertion here.
"""

from __future__ import annotations

import pytest

from tilefoundry import passes
from tilefoundry.ir import core


def test_pass_manager_runs_in_registered_order_and_enforces_requires():
    """The order passes were added is the order they run.

    The order passes were added is the order they run, and a declared
    dependency has to have been added before its dependant.

    There is no topological sort: the manager checks rather than reorders, so a
    pipeline assembled in the wrong order is reported as such instead of being
    quietly repaired into an order nobody wrote down.
    """
    trace: list[str] = []

    class _Traced(passes.ModulePass):
        def __init__(self, tag):
            self.tag = tag
            self.name = tag

        def run(self, module: core.Module) -> core.Module:
            trace.append(self.tag)
            return module

    pm = passes.PassManager()
    pm.add(_Traced("a")).add(_Traced("b")).add(_Traced("c"))

    pm.run(core.Module(name="m", functions=(), entry="x"))
    assert trace == ["a", "b", "c"]

    ran: list[str] = []

    class _A(passes.ModulePass):
        name = "a"

        def run(self, module):
            ran.append("a")
            return module

    class _B(passes.ModulePass):
        name = "b"
        requires = ("a",)

        def run(self, module):
            ran.append("b")
            return module

    empty = core.Module(name="m", functions=(), entry="x")
    assert passes.PassManager(passes=[_A(), _B()]).run(empty) is empty
    assert ran == ["a", "b"]

    with pytest.raises(ValueError, match="requires 'a' not registered before it"):
        passes.PassManager(passes=[_B(), _A()]).run(empty)
    assert ran == ["a", "b"]
