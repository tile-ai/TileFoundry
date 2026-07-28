"""Coverage for tilefoundry.passes — Pass / PassManager orchestration.

Per-transform behaviour lives in the transform's own test file, and the default
``tilefoundry.lower`` pipeline is exercised end to end by every test that
compiles and runs a kernel: a pipeline that stopped wiring itself up cannot
produce CUDA at all, so it needs no separate assertion here.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core.module import Module
from tilefoundry.passes import ModulePass, PassManager


def test_pass_manager_runs_in_registered_order_and_enforces_requires():
    """The order passes were added is the order they run, and a declared
    dependency has to have been added before its dependant.

    There is no topological sort: the manager checks rather than reorders, so a
    pipeline assembled in the wrong order is reported as such instead of being
    quietly repaired into an order nobody wrote down.
    """
    trace: list[str] = []

    class _Traced(ModulePass):
        def __init__(self, tag):
            self.tag = tag
            self.name = tag

        def run(self, module: Module) -> Module:
            trace.append(self.tag)
            return module

    pm = PassManager()
    pm.add(_Traced("a")).add(_Traced("b")).add(_Traced("c"))
    # entry_function() is not invoked by empty PassManager runs.
    pm.run(Module(name="m", functions=(), entry="x"))
    assert trace == ["a", "b", "c"]

    class _A(ModulePass):
        name = "a"

        def run(self, module):
            return module

    class _B(ModulePass):
        name = "b"
        requires = ("a",)

        def run(self, module):
            return module

    PassManager(passes=[_A(), _B()])._check_requires()  # should not raise

    wrong = PassManager(passes=[_B(), _A()])
    with pytest.raises(ValueError, match="requires 'a' not registered before it"):
        wrong._check_requires()
