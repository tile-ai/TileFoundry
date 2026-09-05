"""Effect-form TIR Op for the CUDA bulk asynchronous copy (TMA).

``CopyAsync`` is ``cp.async``: every thread issues its own load and a commit
closes the group. This is ``cp.async.bulk`` -- **one** thread issues a whole
run, an mbarrier signals completion -- and is not a tier of the other. Only the
rank-1 form is defined; the tensor forms take a host-encoded ``TensorMap`` in
place of a size, a different operand list rather than a tier. **Defined but not
lowered**: no CUDA emit; the implementation is in ``ops/tma.cuh``.

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

__all__ = ["TmaBulkCopy"]

_GRAIN_BYTES = 16


@register_op(dialect="T", category="async", name="tma_bulk_copy")
class TmaBulkCopy(Op):
    """Stage a contiguous run from global to shared memory as one bulk copy.

    Exactly one thread issues it and nothing blocks: the copy is still in flight
    when the issuing thread reaches the next statement. Completion lands on
    ``barrier``'s current phase, so the issuing thread pairs this with an
    ``MBarrierArriveExpectTx`` naming the same byte count and every consumer
    waits with ``MBarrierWaitParity``.
    """

    src = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)
    barrier = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(TmaBulkCopy)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(TmaBulkCopy)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    bar = ctx.type_of(call.args[2])
    if src.storage != StorageKind.GMEM:
        ctx.error(call, f"TmaBulkCopy source must be gmem, got {src.storage}")
    if dst.storage != StorageKind.SMEM:
        ctx.error(call, f"TmaBulkCopy destination must be smem, got {dst.storage}")
    if bar.storage != StorageKind.SMEM:
        ctx.error(call, f"TmaBulkCopy barrier must be smem, got {bar.storage}")
    if src.dtype != dst.dtype:
        ctx.error(call, f"TmaBulkCopy dtype mismatch: {src.dtype} vs {dst.dtype}")
    _verify_grain(ctx, call, src, dst)


def _verify_grain(ctx, call, src, dst) -> None:
    """Report unless the transfer is a whole number of ``_GRAIN_BYTES`` grains.

    That constant is the instruction's transfer grain: both addresses and the
    byte count are constrained to it, and off the grain the instruction has no
    defined behaviour, so this checks rather than rounds. Only static extents
    can be checked here; a run whose length is a DimVar carries the same
    constraint to whoever supplies the value.
    """
    if src.shape != dst.shape:
        ctx.error(call, f"TmaBulkCopy shape mismatch: {src.shape} vs {dst.shape}")
        return
    if any(not isinstance(d, int) for d in dst.shape):
        return
    width = getattr(dst.dtype, "bit_width", None)
    if not isinstance(width, int):
        return
    total_bits = width
    for d in dst.shape:
        total_bits *= d
    if total_bits % (_GRAIN_BYTES * 8):
        ctx.error(
            call,
            f"TmaBulkCopy transfer must be a multiple of {_GRAIN_BYTES} bytes, "
            f"got {total_bits // 8}",
        )
