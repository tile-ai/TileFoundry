"""Mesh-scoped barriers and memory fences, without tensor tokens."""
from enum import Enum

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.register import register_op, register_type
from tilefoundry.ir.pattern import LayoutPattern, OrPattern, matched
from tilefoundry.ir.pattern import predicates as P
from tilefoundry.ir.pattern.utils import thread_execution_mesh
from tilefoundry.ir.types.layout import ComposedLayout, size
from tilefoundry.ir.types.layout_algebra import cosize
from tilefoundry.ir.types.mesh import separate
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    identity_access,
    iterating,
    register_access_relation,
)

from .common import attribute, cpu_args, infer
from .memory import AtomicOrder


@register_type(dialect="T", namespace="metal")
class MemoryFence(Enum):
    NONE = "none"
    THREADGROUP = "threadgroup"
    DEVICE = "device"
    ALL = "all"

@register_op(dialect="T", category="sync", namespace="metal", name="barrier")
class Barrier(Op):
    """A whole threadgroup or aligned complete SIMD32 group participates uniformly.

    DEVICE fences device-memory accesses; it does not synchronize threadgroups.
    """
    capability = OpCapability("metal_barrier")
    execution_mesh = thread_execution_mesh()
    fence = attribute(MemoryFence, OrPattern(*MemoryFence), MemoryFence.THREADGROUP)

@register_type(dialect="T", namespace="metal")
class ThreadScope(Enum):
    SYSTEM = "system"

@register_op(dialect="T", category="sync", namespace="metal", name="fence")
class Fence(Op):
    """MLX Metal4 system seq_cst device-memory fence; no execution rendezvous."""
    capability = OpCapability("metal_fence_system")
    execution_mesh = thread_execution_mesh()
    fence = attribute(MemoryFence, MemoryFence.DEVICE, MemoryFence.DEVICE)
    order = attribute(AtomicOrder, AtomicOrder.SEQ_CST, AtomicOrder.SEQ_CST)
    scope = attribute(ThreadScope, ThreadScope.SYSTEM, ThreadScope.SYSTEM)

@register_verify_stmt(Barrier)
def barrier(call, ctx):
    mesh = ctx.current_mesh
    threads = None if mesh is None else next((m for m in separate(mesh) if m.topologies[0].name == "thread"), None)
    if threads is None:
        ctx.error(call, "barrier requires a thread mesh")
    layout = threads.layout
    offset = layout.offset if isinstance(layout, ComposedLayout) else 0
    layout = layout.outer if isinstance(layout, ComposedLayout) else layout
    count, extent = size(layout), threads.topologies[0].size
    full = offset == 0 and count == extent
    simd = count == 32 and isinstance(offset, int) and offset % 32 == 0
    if not (full or simd) or offset < 0 or offset + count > extent or cosize(layout)!=count:
        ctx.error(call, "barrier requires a whole threadgroup or an aligned complete SIMD32 group")
    if matched(LayoutPattern(predicates=(P.Forward(), P.Injective())), layout) is None:
        ctx.error(call, "barrier participants must be forward and injective")

@register_verify_stmt(Fence)
def fence(call, ctx):
    """An operand-free fence needs an execution scope, but no group rendezvous."""
    if ctx.current_mesh is None or not any(t.name == "thread" for t in ctx.current_mesh.topologies):
        ctx.error(call, "system fence requires a thread execution scope")


def sync_access(call, ctx):
    return iterating((), (identity_access(0),))

def eval_sync(ctx):
    cpu_args(ctx)
    return None

register_typeinfer(Barrier)(infer)
register_typeinfer(Fence)(infer)
register_access_relation(Barrier)(sync_access)
register_access_relation(Fence)(sync_access)
register_schedule_eval(Barrier)(eval_sync)
register_schedule_eval(Fence)(eval_sync)
