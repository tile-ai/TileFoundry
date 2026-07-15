"""Emitter for the ``tir.Sync`` op — emits the uniform runtime barrier call.

The barrier kind and participant geometry come from ``classify`` /
``participation`` (shared with verify); the emit passes them as template
parameters to ``tilefoundry::ops::sync``.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.sync import Sync, SyncBarrier, classify, participation

_SYNC = "tilefoundry::ops::sync"
_KIND = "tilefoundry::ops::SyncKind"


@register_codegen_cuda(Sync)
def _emit(call, ctx: CodegenContext) -> None:
    mesh = call.target.mesh
    barrier = classify(mesh)

    # Emit the uniform ``sync<Kind, ...>`` entry: the barrier kind and the
    # codegen-static participant geometry go as template parameters.
    if barrier is SyncBarrier.GRID:
        # The grid counter pair is defined once per module (internal linkage);
        # see the module template.
        ctx.emit(f"{_SYNC}<{_KIND}::grid>(tilefoundry::tf_grid_bar_state);")
        return

    p = participation(mesh)

    if barrier is SyncBarrier.SYNCTHREADS:
        ctx.emit(f"{_SYNC}<{_KIND}::syncthreads>();")
        return

    if barrier is SyncBarrier.SYNCWARP:
        if p.full_cta:
            # The whole block is a single warp — every lane participates.
            ctx.emit(f"{_SYNC}<{_KIND}::syncwarp_full>();")
            return
        # A contiguous lane subset of one warp: the runtime predicate keeps
        # non-participant lanes out of the masked warp sync.
        ctx.emit(
            f"{_SYNC}<{_KIND}::syncwarp_masked, {p.base}, {p.count}, "
            f"0x{p.lane_mask:08x}u>();"
        )
        return

    # BAR_SYNC: a warp-aligned multi-warp subset uses a named barrier; the
    # runtime participant predicate keeps non-participants out of it.
    bid = ctx.alloc_barrier_id()
    ctx.emit(
        f"{_SYNC}<{_KIND}::bar_sync, {p.base}, {p.count}, 0u, {bid}>();"
    )
