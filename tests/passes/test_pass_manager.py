"""Coverage for tilefoundry.passes — Pass / PassManager orchestration + the
default ``tilefoundry.lower`` pipeline (per-transform behavior lives in the
transform's own test file, e.g. ``test_hir_to_tir.py``)."""

from __future__ import annotations

import pytest

import tilefoundry
from tests.fixtures.demo_ir import build_demo
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.passes import ModulePass, PassManager

# ---------------------------------------------------------------------------
# PassManager.add + ordered run
# ---------------------------------------------------------------------------

def test_pass_manager_runs_in_registered_order():
    trace: list[str] = []

    class _P(ModulePass):
        def __init__(self, tag):
            self.tag = tag
            self.name = tag

        def run(self, module: Module) -> Module:
            trace.append(self.tag)
            return module

    pm = PassManager()
    pm.add(_P("a")).add(_P("b")).add(_P("c"))
    module = Module(name="m", functions=(), entry="x")
    # entry_function() is not invoked by empty PassManager runs.
    pm.run(module)
    assert trace == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# `requires` enforcement — order assert only (no topological sort in MVP).
# ---------------------------------------------------------------------------

def test_pass_manager_requires_enforces_prior_pass_seen():
    class _A(ModulePass):
        name = "a"

        def run(self, module):
            return module

    class _B(ModulePass):
        name = "b"
        requires = ("a",)

        def run(self, module):
            return module

    ok = PassManager(passes=[_A(), _B()])
    ok._check_requires()  # should not raise

    wrong = PassManager(passes=[_B(), _A()])
    with pytest.raises(ValueError, match="requires 'a' not registered before it"):
        wrong._check_requires()


# ---------------------------------------------------------------------------
# tilefoundry.compile top-level wires the default pipeline.
# ---------------------------------------------------------------------------

def test_tilefoundry_lower_drives_default_pipeline():

    fn, _, _ = build_demo()
    mod = Module(name="main", functions=(fn,), entry=fn.name)
    result = tilefoundry.lower(mod, target="cuda")
    [out_fn] = result.functions
    assert isinstance(out_fn, PrimFunction)
    assert result.entry == "demo"


