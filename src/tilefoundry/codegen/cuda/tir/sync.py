"""Emitter for the ``tir.Sync`` op — emits the mesh-scoped runtime barrier.

The mesh is the whole call. Which barrier runs is ``ops::sync``'s own answer,
read off the ``Mesh`` type's scope, size and base, so the emitted line names who
must agree and never which instruction does it — and a mesh reshaped upstream
cannot leave a stale barrier behind at the call site.

``classify`` still runs here for its refusals: a partial grid sync and a ragged
cross-warp subset are deadlocks, and a ``VerifyError`` before codegen says so
better than the ``static_assert`` that backs it up inside nvcc.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.tir.stmts.mesh_scope import mesh_type
from tilefoundry.ir.tir.sync import Sync, SyncBarrier, classify
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen

_SYNC = "tilefoundry::ops::sync"


def _mesh_value(mesh, ctx: CudaCodegenContext) -> str:
    """*mesh* as a C++ value, through the enclosing scope's alias where it fits."""
    entry = ctx._mesh_aliases.get(id(mesh))
    if entry is not None:
        return f"{entry[0]}{{}}"
    inline = mesh_type(mesh)
    for alias_name, type_str in ctx._mesh_aliases.values():
        if type_str == inline:
            return f"{alias_name}{{}}"
    return f"{inline}{{}}"


@register_codegen(CudaTarget, Role.EMIT, Sync)
def _emit(call, ctx: CudaCodegenContext) -> None:
    """Emit the barrier as the mesh it covers, plus whatever that tier needs."""
    mesh = call.target.mesh
    barrier = classify(mesh)
    value = _mesh_value(mesh, ctx)
    if barrier is SyncBarrier.GRID:
        ctx.needs_grid_barrier_state = True
        ctx.emit(f"{_SYNC}({value}, tilefoundry_grid_bar_state);")
        return
    if barrier is SyncBarrier.BAR_SYNC:
        bid = ctx.alloc_barrier_id()
        ctx.emit(f"{_SYNC}({value}, tilefoundry::ops::bar_id<{bid}>);")
        return
    ctx.emit(f"{_SYNC}({value});")
