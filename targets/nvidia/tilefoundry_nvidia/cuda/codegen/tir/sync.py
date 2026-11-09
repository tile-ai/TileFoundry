"""Emitter for the ``tir.sync.Sync`` op — emits the mesh-scoped runtime barrier.

The mesh is the whole call. Which barrier runs is ``ops::sync``'s own answer,
read off the ``Mesh`` type's scope, size and base, so the emitted line names who
must agree and never which instruction does it — and a mesh reshaped upstream
cannot leave a stale barrier behind at the call site.

``classify`` still runs here for its refusals: a partial grid sync and a ragged
cross-warp subset are deadlocks, and a ``VerifyError`` before codegen says so
better than the ``static_assert`` that backs it up inside nvcc.
"""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir import tir
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context

from .stmts import mesh_type

SYNC = "tilefoundry::ops::sync"


def mesh_value(mesh, ctx: context.Context) -> str:
    """*mesh* as a C++ value, through the enclosing scope's alias where it fits."""
    entry = ctx.mesh_aliases.get(id(mesh))
    if entry is not None:
        return f"{entry[0]}{{}}"
    inline = mesh_type(mesh)
    for alias_name, type_str in ctx.mesh_aliases.values():
        if type_str == inline:
            return f"{alias_name}{{}}"
    return f"{inline}{{}}"


@codegen.register(target.Target, codegen.Role.EMIT, tir.sync.Sync)
def emit(call, ctx: context.Context) -> None:
    """Emit the barrier as the mesh it covers, plus whatever that tier needs."""
    mesh = call.target.mesh
    barrier = tir.sync.classify(mesh)
    value = mesh_value(mesh, ctx)
    if barrier is tir.sync.SyncBarrier.GRID:
        ctx.require_grid_barrier_state()
        ctx.emit(f"{SYNC}({value}, tilefoundry_grid_bar_state);")
        return
    if barrier is tir.sync.SyncBarrier.BAR_SYNC:
        bid = ctx.alloc_barrier_id()
        ctx.emit(f"{SYNC}({value}, tilefoundry::ops::bar_id<{bid}>);")
        return
    ctx.emit(f"{SYNC}({value});")


__all__: list[str] = []
