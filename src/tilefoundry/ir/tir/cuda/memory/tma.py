"""Effect-form TIR Op for a barrier-completing gmem→smem staging copy.

``CopyAsync`` is ``cp.async``: the thread that issues is the thread that waits.
This completes on an mbarrier, which is what lets a consumer wait for a tile it
did not fetch. Which instruction carries it is the runtime's choice from the
operand shard layouts, not a tier this op names.

**Defined but not lowered**: no CUDA emit; the implementation is in
``ops/tma.cuh`` ([runtime §3](docs/spec/runtime.md#3-runtime-ops)).

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

__all__ = ["TmaCopy"]


@register_op(dialect="T", category="async", name="tma_copy")
class TmaCopy(Op):
    """Stage a tile from global to shared memory, completing on a barrier.

    Nothing blocks: the copy may still be in flight when the issuing thread
    reaches the next statement. Completion lands on ``barrier``'s current
    phase, so every consumer waits with ``MBarrierWaitParity``. The arrival
    that declares the transferred bytes belongs to the implementation, which
    issues it on the same instruction as the copy; a caller that arrived
    separately would be declaring a count the op already knows.
    """

    src = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)
    barrier = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(TmaCopy)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(TmaCopy)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    bar = ctx.type_of(call.args[2])
    if src.storage != StorageKind.GMEM:
        ctx.error(call, f"TmaCopy source must be gmem, got {src.storage}")
    if dst.storage != StorageKind.SMEM:
        ctx.error(call, f"TmaCopy destination must be smem, got {dst.storage}")
    if bar.storage != StorageKind.SMEM:
        ctx.error(call, f"TmaCopy barrier must be smem, got {bar.storage}")
    if src.dtype != dst.dtype:
        ctx.error(call, f"TmaCopy dtype mismatch: {src.dtype} vs {dst.dtype}")
    if src.shape != dst.shape:
        ctx.error(call, f"TmaCopy shape mismatch: {src.shape} vs {dst.shape}")
