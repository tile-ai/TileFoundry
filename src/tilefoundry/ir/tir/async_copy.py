"""Effect-form TIR Ops for asynchronous (``cp.async``) gmem→smem staging."""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer, register_verify_stmt
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.storage import StorageKind


@register_op(dialect="T", category="async", name="copy_async")
class CopyAsync(Op):
    """Async gmem→smem copy (``cp.async.cg.shared.global``); non-blocking."""
    source = ParamDef(kind="input", pattern=Tensor)
    destination = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(CopyAsync)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(CopyAsync)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if dst.storage != StorageKind.SMEM:
        ctx.error(call, f"CopyAsync destination must be smem, got {dst.storage}")
    if src.storage != StorageKind.GMEM:
        ctx.error(call, f"CopyAsync source must be gmem, got {src.storage}")
    if src.dtype != dst.dtype:
        ctx.error(call, f"CopyAsync dtype mismatch: {src.dtype} vs {dst.dtype}")


@register_op(dialect="T", category="async", name="cp_async_commit")
class CpAsyncCommit(Op):
    """Close the current in-flight ``cp.async`` group (``commit_group``)."""


@register_typeinfer(CpAsyncCommit)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(CpAsyncCommit)
def _(call: "Call", ctx: "VerifyContext") -> None:
    return None


@register_op(dialect="T", category="async", name="cp_async_wait")
class CpAsyncWait(Op):
    """Block until all but the ``n`` newest committed groups have arrived (``cp.async.wait_group n``)."""
    n = ParamDef(kind="attribute", annotation=int, default=0)


@register_typeinfer(CpAsyncWait)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(CpAsyncWait)
def _(call: "Call", ctx: "VerifyContext") -> None:
    n = call.target.n
    if not isinstance(n, int) or n < 0:
        ctx.error(call, f"CpAsyncWait.n must be a non-negative int, got {n!r}")


__all__ = ["CopyAsync", "CpAsyncCommit", "CpAsyncWait"]
