"""Effect-form TIR Ops that order wgmma and async proxy accesses.

[tir](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/tir.md#wgmma-group-and-proxy-fence-ops).
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import WildcardPattern, utils
from tilefoundry.ir.tir.sync import participation
from tilefoundry.ir.types import Mesh, StorageKind, UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)
from tilefoundry_nvidia.cuda.ir.tir.nn.wgmma import WARPGROUP

__all__ = ["FenceProxyAsync", "WgmmaCommitGroup", "WgmmaFence", "WgmmaWaitGroup", "whole_groups"]


@register_op(dialect="T", category="sync", name="wgmma_fence")
class WgmmaFence(Op):
    """Order earlier register accesses before the warpgroup's next wgmma."""

    execution_mesh = utils.thread_execution_mesh()


@register_op(dialect="T", category="sync", name="wgmma_commit_group")
class WgmmaCommitGroup(Op):
    """Close the warpgroup's uncommitted wgmmas into one group."""

    execution_mesh = utils.thread_execution_mesh()


@register_op(dialect="T", category="sync", name="wgmma_wait_group")
class WgmmaWaitGroup(Op):
    """Wait until all but the ``n`` newest committed wgmma groups complete."""

    execution_mesh = utils.thread_execution_mesh()

    n = ParamDef(
        kind="attribute",
        annotation=int,
        pattern=WildcardPattern("wait_groups", predicates=(WildcardPattern("wait_groups") >= 0,)),
        default=0,
    )


@register_op(dialect="T", category="sync", name="fence_proxy_async")
class FenceProxyAsync(Op):
    """Make this thread's earlier ``storage`` accesses visible to the async proxy.

    ``storage`` is global memory, or None for shared memory: the two the async
    proxy reaches.
    """

    execution_mesh = utils.thread_execution_mesh()

    storage = ParamDef(kind="attribute", annotation=StorageKind, optional=True, default=None)


def whole_groups(mesh: Mesh, size: int) -> bool:
    """Whether ``mesh``'s threads are whole, aligned groups of ``size`` threads.

    It reads which threads the mesh holds, not how its layout orders them: a
    ``Layout((256,), (1,))`` thread mesh is two warpgroups here, while
    ``execution_mesh_pattern(WARPGROUP, groups=True)`` matches only a layout
    whose every group is laid out as ``WARPGROUP``.
    """
    held = participation(mesh)
    return held.base % size == 0 and held.count % size == 0


def _verify_warpgroups(call, ctx) -> None:
    """Admit a wgmma group step only where whole, aligned warpgroups run it together.

    The threads are those of the innermost mesh scope, whatever CTAs run it.
    """
    mesh = ctx.mesh_scope[-1] if ctx.mesh_scope else None
    if mesh is None or not whole_groups(mesh, participation(WARPGROUP).count):
        ctx.error(call, f"{type(call.target).__name__} needs whole warpgroups, got {mesh}")


@register_verify_stmt(FenceProxyAsync)
def _verify_proxy_storage(call, ctx) -> None:
    storage = call.target.storage
    if storage not in (None, StorageKind.GMEM):
        ctx.error(call, f"FenceProxyAsync storage is gmem, or None for smem; got {storage}")


for _op in (WgmmaFence, WgmmaCommitGroup, WgmmaWaitGroup, FenceProxyAsync):
    register_typeinfer(_op)(lambda call, ctx: UnitType())
    register_access_relation(_op)(identity_relations)
for _op in (WgmmaFence, WgmmaCommitGroup, WgmmaWaitGroup):
    register_verify_stmt(_op)(_verify_warpgroups)
