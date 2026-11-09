"""Effect-form TIR Ops that order wgmma and async proxy accesses.

[tir](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/tir.md#wgmma-group-and-proxy-fence-ops).
"""

from __future__ import annotations

from tilefoundry.interfaces import access, typeinfer, verify
from tilefoundry.ir import expr, pattern, tir, types
from tilefoundry_nvidia.cuda.ir.tir import nn

__all__ = ["FenceProxyAsync", "WgmmaCommitGroup", "WgmmaFence", "WgmmaWaitGroup", "whole_groups"]


@expr.register_op(dialect="T", category="sync", name="wgmma_fence")
class WgmmaFence(expr.Op):
    """Order earlier register accesses before the warpgroup's next wgmma."""

    execution_mesh = pattern.thread_execution_mesh()


@expr.register_op(dialect="T", category="sync", name="wgmma_commit_group")
class WgmmaCommitGroup(expr.Op):
    """Close the warpgroup's uncommitted wgmmas into one group."""

    execution_mesh = pattern.thread_execution_mesh()


@expr.register_op(dialect="T", category="sync", name="wgmma_wait_group")
class WgmmaWaitGroup(expr.Op):
    """Wait until all but the ``n`` newest committed wgmma groups complete."""

    execution_mesh = pattern.thread_execution_mesh()

    n = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("wait_groups", predicates=(pattern.Wildcard("wait_groups") >= 0,)),
        default=0,
    )


@expr.register_op(dialect="T", category="sync", name="fence_proxy_async")
class FenceProxyAsync(expr.Op):
    """Make this thread's earlier ``storage`` accesses visible to the async proxy.

    ``storage`` is global memory, or None for shared memory: the two the async
    proxy reaches.
    """

    execution_mesh = pattern.thread_execution_mesh()

    storage = expr.ParamDef(
        kind="attribute", annotation=expr.StorageKind, optional=True, default=None
    )


def whole_groups(mesh: types.Mesh, size: int) -> bool:
    """Whether ``mesh``'s threads are whole, aligned groups of ``size`` threads.

    It reads which threads the mesh holds, not how its layout orders them: a
    ``Layout((256,), (1,))`` thread mesh is two warpgroups here, while
    ``execution_mesh_pattern(WARPGROUP, groups=True)`` matches only a layout
    whose every group is laid out as ``WARPGROUP``.
    """
    held = tir.sync.participation(mesh)
    return held.base % size == 0 and held.count % size == 0


def verify_warpgroups(call, ctx) -> None:
    """Admit a wgmma group step only where whole, aligned warpgroups run it together.

    The threads are those of the innermost mesh scope, whatever CTAs run it.
    """
    mesh = ctx.mesh_scope[-1] if ctx.mesh_scope else None
    if mesh is None or not whole_groups(mesh, tir.sync.participation(nn.WARPGROUP).count):
        ctx.error(call, f"{type(call.target).__name__} needs whole warpgroups, got {mesh}")


@verify.register(FenceProxyAsync)
def verify_proxy_storage(call, ctx) -> None:
    storage = call.target.storage
    if storage not in (None, expr.StorageKind.GMEM):
        ctx.error(call, f"FenceProxyAsync storage is gmem, or None for smem; got {storage}")


for op in (WgmmaFence, WgmmaCommitGroup, WgmmaWaitGroup, FenceProxyAsync):
    typeinfer.register(op)(lambda call, ctx: types.UnitType())
    access.register_relation(op)(access.identity_relations)
for op in (WgmmaFence, WgmmaCommitGroup, WgmmaWaitGroup):
    verify.register(op)(verify_warpgroups)
