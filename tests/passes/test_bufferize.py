"""BufferizePass — trivial-policy gate.

policy gives every logical buffer its own physical allocation, so the
pass leaves the ``PrimFunction`` body structurally unchanged. What is worth
asserting beyond that is the traversal: a collector that misses a buffer does not
fail, it silently leaves an allocation out of the schedule it is supposed to
place, and the arm most easily missed is the one a hand-rolled walk does not know
about.
"""

from __future__ import annotations

from tests.fixtures.placed.rmsnorm import RmsnormModule
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.memory import AllocTensor as AllocTensorOp
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Abort, LetStmt, Sequential
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.passes.transforms import BufferizePass, HirToTirPass
from tilefoundry.passes.transforms.bufferize import LifetimeCollector


def _lower() -> tuple[PrimFunction, Module]:
    fn = RmsnormModule.entry_function()
    module = Module(name="t", functions=(fn,), entry=fn.name)
    module = HirToTirPass().run(module)
    [pf] = module.functions
    assert isinstance(pf, PrimFunction)
    return pf, module


def test_bufferize_returns_module_unchanged():
    pf_before, module = _lower()
    new_module = BufferizePass().run(module)
    [pf_after] = new_module.functions

    assert pf_after is pf_before


def test_lifetime_collector_finds_buffer_inside_dispatch_call_fallback():
    """A buffer allocated inside a ``DispatchCall``'s ``fallback`` arm must be collected.

    A buffer allocated inside a ``DispatchCall``'s ``fallback`` arm must
    be collected. A hand-rolled Stmt walk without ``DispatchCall`` coverage
    silently skips it ([visitor-mutator §1](docs/spec/visitor-mutator.md#1-role)).
    """
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
