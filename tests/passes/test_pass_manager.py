"""Coverage for tilefoundry.passes — Pass / PassManager + the two MVP transforms."""

from __future__ import annotations

import pytest

import tilefoundry
from tests.models.demo.demo_ir import build_demo
from tilefoundry.dump import DumpFlags, DumpScope, MemoryDumper
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.passes import ModulePass, PassManager
from tilefoundry.passes.transforms import HirToTirPass


def _demo_module() -> Module:
    fn, _, _ = build_demo()
    return Module(name="t", functions=(fn,), entry=fn.name)


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
# Per-pass dump emits before/after IR through DumpScope.
# ---------------------------------------------------------------------------

def test_pass_manager_emits_per_pass_before_after_dump():

    class _NoOp(ModulePass):
        def __init__(self, tag):
            self.name = tag

        def run(self, module):
            return module

    pm = PassManager()
    pm.add(_NoOp("first")).add(_NoOp("second")).add(_NoOp("third"))

    dumper = MemoryDumper()
    with DumpScope(dumper=dumper, flags=DumpFlags.ALL):
        pm.run(Module(name="m", functions=(), entry="x"))

    keys = sorted(dumper.entries.keys())
    assert keys == [
        "00_first/after.txt",
        "00_first/before.txt",
        "01_second/after.txt",
        "01_second/before.txt",
        "02_third/after.txt",
        "02_third/before.txt",
    ]


# ---------------------------------------------------------------------------
# HirToTirPass replaces the hir.Function with a tir.PrimFunction.
# ---------------------------------------------------------------------------

def test_hir_to_tir_pass_lowers_module():
    module = _demo_module()
    p = HirToTirPass()
    new_module = p.run(module)

    assert new_module is not module
    assert new_module.entry == module.entry
    [fn] = new_module.functions
    assert isinstance(fn, PrimFunction)
    assert fn.name == "demo"


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


def test_tilefoundry_lower_auto_derives_meshes():
    """lower() auto-derives meshes from the HIR body, no explicit kwargs needed."""

    fn, _, _ = build_demo()
    mod = Module(name="main", functions=(fn,), entry=fn.name)
    result = tilefoundry.lower(mod, target="cuda")
    [out_fn] = result.functions
    assert isinstance(out_fn, PrimFunction)

