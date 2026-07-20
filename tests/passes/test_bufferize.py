"""BufferizePass — trivial-policy gate.

policy gives every logical buffer its own physical allocation, so the
pass leaves the ``PrimFunction`` body structurally unchanged.
"""
from __future__ import annotations

from tests.fixtures.demo_ir import build_demo
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.memory import AllocTensor as AllocTensorOp
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Abort, LetStmt, Sequential
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.storage import StorageKind
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


def test_lifetime_collector_finds_buffer_inside_dispatch_call_fallback():
    """A buffer allocated inside a ``DispatchCall``'s ``fallback`` arm must
    be collected. A hand-rolled Stmt walk without ``DispatchCall`` coverage
    silently skips it (docs/spec/visitor-mutator.md §1)."""
    buf_type = TensorType(shape=(4,), dtype=DType.f32, layout=None, storage=StorageKind.RMEM)
    buf_var = Var(type=buf_type, name="buf")
    alloc_call = Call(type=buf_type, target=AllocTensorOp(tensor_type=buf_type), args=())
    fallback = Sequential(
        body=(LetStmt(var=buf_var, value=alloc_call, body=Sequential(body=(Abort(),))),)
    )
    dispatch = DispatchCall(
        callee_name="f",
        subjects=(),
        case_patterns=(),
        case_calls=(),
        fallback=fallback,
    )
    pf = PrimFunction(name="f", params=(), body=Sequential(body=(dispatch,)))

    entries = LifetimeCollector().collect(pf)

    assert [e.var.name for e in entries] == ["buf"]
