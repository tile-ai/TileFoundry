"""BufferizePass — trivial-policy gate.

policy gives every logical buffer its own physical allocation, so the
pass leaves the ``PrimFunction`` body structurally unchanged.
"""
from __future__ import annotations

from tests.fixtures.demo_ir import build_demo
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.memory import AllocTensor as AllocTensorOp
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import LetStmt, Sequential
from tilefoundry.passes.transforms import BufferizePass, HirToTirPass
from tilefoundry.passes.transforms.bufferize import (
    BufferEntry,
    BufferScheduler,
    LifetimeCollector,
    Placement,
)


def _lower() -> tuple[PrimFunction, Module]:
    fn, _, _ = build_demo()
    module = Module(name="t", functions=(fn,), entry=fn.name)
    module = HirToTirPass().run(module)
    [pf] = module.functions
    assert isinstance(pf, PrimFunction)
    return pf, module


def _collect_alloc_vars(pf: PrimFunction) -> list[str]:
    names: list[str] = []

    def walk(stmt) -> None:
        if isinstance(stmt, LetStmt):
            if isinstance(stmt.value, Call) and isinstance(
                stmt.value.target, AllocTensorOp
            ):
                names.append(stmt.var.name)
            walk(stmt.body)
            return
        if isinstance(stmt, Sequential):
            for s in stmt.body:
                walk(s)
            return
        # Non-binding stmt — descend into any Sequential body field.
        body = getattr(stmt, "body", None)
        if isinstance(body, Sequential):
            walk(body)

    walk(pf.body)
    return names


def test_bufferize_returns_module_unchanged():
    pf_before, module = _lower()
    new_module = BufferizePass().run(module)
    [pf_after] = new_module.functions
    # Trivial policy: each logical buffer keeps its own AllocTensor; IR
    # identity is preserved (PrimFuncPass returns the same fn object).
    assert pf_after is pf_before


def test_bufferize_preserves_alloc_chain_and_params():
    pf_before, module = _lower()
    new_module = BufferizePass().run(module)
    [pf_after] = new_module.functions
    assert pf_after.params == pf_before.params
    assert _collect_alloc_vars(pf_after) == _collect_alloc_vars(pf_before)


def test_lifetime_collector_emits_one_entry_per_alloc():
    pf, _ = _lower()
    entries = LifetimeCollector().collect(pf)
    assert all(isinstance(e, BufferEntry) for e in entries)
    names = [e.var.name for e in entries]
    assert names == _collect_alloc_vars(pf)


def test_scheduler_assigns_independent_pool_per_buffer():
    pf, _ = _lower()
    entries = LifetimeCollector().collect(pf)
    placements = BufferScheduler().schedule(entries)
    assert len(placements) == len(entries)
    assert all(isinstance(p, Placement) for p in placements)
    assert all(p.offset == 0 for p in placements)
    # Trivial policy: pool_id is the buffer's own var → no pool sharing.
    pool_ids = {id(p.pool_id) for p in placements}
    assert len(pool_ids) == len(placements)
