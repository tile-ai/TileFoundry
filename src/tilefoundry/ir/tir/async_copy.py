"""Effect-form TIR Ops for asynchronous (``cp.async``) gmem→smem staging."""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    DistinctConstraint,
    SameModesConstraint,
    any_threads,
    moved_tile,
    vector,
)
from tilefoundry.ir.tir.verify import verify_between, verify_operands
from tilefoundry.ir.types import Layout, UnitType
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt


@register_op(dialect="T", category="async", name="copy_async")
class CopyAsync(Op):
    """Async gmem→smem copy (``cp.async.cg.shared.global``); non-blocking."""

    src = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        pattern=moved_tile(0, S.GMEM, vector(0)),
    )
    dst = ParamDef(
        kind="input",
        effect=MemoryEffect.WRITE,
        pattern=moved_tile(1, S.SMEM, vector(1)),
    )
    between = (
        DistinctConstraint("storage", "src", "dst"),
        SameModesConstraint("src", "dst"),
    )
    smem_layout = ParamDef(
        kind="attribute",
        annotation=Layout,
        optional=True,
        default=None,
    )
    scope = any_threads()


@register_typeinfer(CopyAsync)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(CopyAsync)
def verify_copy_async(call: "Call", ctx: "VerifyContext") -> None:
    """Run native storage checks before layout relations and vector patterns."""
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if dst.storage != S.SMEM:
        ctx.error(call, f"CopyAsync destination must be smem, got {dst.storage}")
    if src.storage != S.GMEM:
        ctx.error(call, f"CopyAsync source must be gmem, got {src.storage}")
    if src.dtype != dst.dtype:
        ctx.error(call, f"CopyAsync dtype mismatch: {src.dtype} vs {dst.dtype}")
    verify_between(call, ctx)
    verify_operands(call, ctx, "copy_async")


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
    """Wait until all but the ``n`` newest committed async groups arrive."""

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
