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

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.codegen.cuda.tir.stmts.mesh_scope import mesh_type
from tilefoundry.ir.tir.sync import Sync, SyncBarrier, classify

_SYNC = "tilefoundry::ops::sync"


def _mesh_value(mesh, ctx: CodegenContext) -> str:
    """*mesh* as a C++ value, through the enclosing scope's alias where it fits."""
    entry = ctx._mesh_aliases.get(id(mesh))
    if entry is not None:
        return f"{entry[0]}{{}}"
    inline = mesh_type(mesh)
    for alias_name, type_str in ctx._mesh_aliases.values():
        if type_str == inline:
            return f"{alias_name}{{}}"
    return f"{inline}{{}}"


@register_codegen_cuda(Sync)
def _emit(call, ctx: CodegenContext) -> None:
    """Emit the barrier as the mesh it covers, plus whatever that tier needs."""
    mesh = call.target.mesh
    barrier = classify(mesh)
    value = _mesh_value(mesh, ctx)
    if barrier is SyncBarrier.GRID:
        ctx.needs_grid_barrier_state = True
        ctx.emit(f"{_SYNC}({value}, tilefoundry::tf_grid_bar_state);")
        return
    if barrier is SyncBarrier.BAR_SYNC:
        bid = ctx.alloc_barrier_id()
        ctx.emit(f"{_SYNC}({value}, tilefoundry::ops::bar_id<{bid}>);")
        return
    ctx.emit(f"{_SYNC}({value});")
