"""Emitter for the ``tir.Sync`` op — a mesh-scoped barrier.

The barrier kind is derived from the participating thread set (shared with
verify via ``classify`` / ``participation`` so codegen cannot disagree):

- whole CTA → ``__syncthreads()`` (or ``__syncwarp()`` when the block is one
  warp);
- a contiguous lane subset inside one warp → ``__syncwarp(mask)`` guarded by a
  participant predicate;
- a warp-aligned contiguous multi-warp subset → ``bar.sync <id>, <count>``
  guarded by a participant predicate, with ``<id>`` allocated implicitly per
  kernel.

Non-participant threads never execute the barrier; every participant executes
the same id/count.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.sync import SyncBarrier, Sync, classify, participation

_TID = "tilefoundry::program_id<tilefoundry::TopologyScope::thread>()"


@register_codegen_cuda(Sync)
def _emit(call, ctx: CodegenContext) -> None:
    mesh = call.target.mesh
    barrier = classify(mesh)

    if barrier is SyncBarrier.GRID:
        # Grid-wide software barrier over the module's counter pair. The
        # counter has internal linkage and is defined once per generated
        # module source (see the module template), so a header include never
        # introduces a shared/duplicated global symbol across translation units.
        ctx.emit("tilefoundry::ops::grid_barrier(tilefoundry::tf_grid_bar_state);")
        return

    p = participation(mesh)

    if barrier is SyncBarrier.SYNCTHREADS:
        ctx.emit("__syncthreads();")
        return

    if barrier is SyncBarrier.SYNCWARP:
        if p.full_cta:
            # The whole block is a single warp — every lane participates.
            ctx.emit("__syncwarp();")
            return
        # A contiguous lane subset of one warp: only the participant lanes run
        # the masked warp sync.
        ctx.emit(
            f"if ({_TID} >= {p.base} && {_TID} < {p.base + p.count}) "
            f"__syncwarp(0x{p.lane_mask:08x}u);"
        )
        return

    # BAR_SYNC: a warp-aligned multi-warp subset uses a named barrier; the
    # participant predicate keeps non-participants out of it.
    bid = ctx.alloc_barrier_id()
    ctx.emit(
        f"if ({_TID} >= {p.base} && {_TID} < {p.base + p.count}) "
        f'asm volatile("bar.sync %0, %1;" :: "r"({bid}), "r"({p.count}));'
    )
