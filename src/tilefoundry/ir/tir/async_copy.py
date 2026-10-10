"""Effect-form TIR Ops for asynchronous (``cp.async``) gmem→smem staging."""

from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    DistinctConstraint,
    OrPattern,
    SameModesConstraint,
    TensorPattern,
    utils,
)
from tilefoundry.ir.pattern.pattern import WildcardPattern
from tilefoundry.ir.pattern.predicates import WholeVectors
from tilefoundry.ir.types import DType, Layout, UnitType
from tilefoundry.ir.types.layout import ComposedLayout, Swizzle
from tilefoundry.ir.types.shard_layout import Broadcast, ShardLayout
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    gather_relations,
    identity_relations,
    register_access_relation,
)

ASYNC_WIDTHS = (4, 8, 16)


def is_indexed_copy(args) -> bool:
    """A TIR CopyAsync supplies src, dst, and the optional index input."""
    return len(args) == 3


def is_indexed_schedule(args) -> bool:
    """A HIR CopyAsync schedule supplies only its reads: src and index."""
    return len(args) == 2


class _CopyModes(SameModesConstraint):
    """An indexed transfer's contiguous vector must stay inside one row."""

    def holds(self, operands: dict) -> bool:
        if not super().holds(operands):
            return False
        if "index" not in operands:
            return True
        pair = self.pair(operands)
        if pair is None:
            return True
        return all(reading[0][0] != 0 for reading in self.readings(pair))

    def written(self) -> str:
        return super().written() + "; indexed vectors must not cross axis 0"


def indexed_width(source, destination) -> int:
    """The common row-aligned width already admitted by the vector patterns."""
    widths = tuple(
        set(
            WholeVectors(WildcardPattern("width"), "dtype", ASYNC_WIDTHS).available_widths(
                type_.layout, {"dtype": type_.dtype}
            )
        )
        for type_ in (source, destination)
    )
    shared = widths[0] & widths[1]
    if not shared or not _CopyModes("src", "dst").holds(
        {"src": source, "dst": destination, "index": True}
    ):
        raise ValueError("indexed CopyAsync requires common aligned vectors inside a row")
    return max(shared)


def _indexed_layout(layout) -> bool:
    """Every issuer must see the complete table, destination, and index."""
    if layout is None or isinstance(layout, Layout):
        return True
    if isinstance(layout, ShardLayout):
        return all(isinstance(attr, Broadcast) for attr in layout.attrs) and _indexed_layout(
            layout.layout
        )
    if isinstance(layout, ComposedLayout):
        return _indexed_layout(layout.outer) and _indexed_layout(layout.inner)
    return isinstance(layout, Swizzle)


@register_op(dialect="T", category="async", name="copy_async")
class CopyAsync(Op):
    """Async gmem→smem copy (``cp.async.cg.shared.global``); non-blocking."""

    capability = OpCapability("cp.async")
    execution_mesh = utils.thread_execution_mesh()

    src = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        pattern=utils.operand_tile(
            0,
            S.GMEM,
            utils.whole_vectors(0, ASYNC_WIDTHS),
            execution_mesh,
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=MemoryEffect.WRITE,
        pattern=utils.operand_tile(
            1,
            S.SMEM,
            utils.whole_vectors(1, ASYNC_WIDTHS),
            execution_mesh,
        ),
    )
    between = (
        DistinctConstraint("storage", "src", "dst"),
        _CopyModes("src", "dst"),
    )
    index = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        optional=True,
        default=None,
        pattern=TensorPattern(shape=(None,), dtype=OrPattern(DType.i32, DType.i64)),
    )
    fill = ParamDef(kind="attribute", annotation=float | None, default=None)
    smem_layout = ParamDef(
        kind="attribute",
        annotation=Layout,
        optional=True,
        default=None,
    )


@register_typeinfer(CopyAsync)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_access_relation(CopyAsync)
def _copy_async_access(call, ctx):
    if not is_indexed_copy(call.args):
        return identity_relations(call, ctx)
    source = ctx.type_of(call.args[0])
    index = ctx.type_of(call.args[2])
    source_access, index_access, out = gather_relations(source, index, 0)
    return source_access, out, index_access, out


@register_schedule_eval(CopyAsync)
def _eval_scheduled_copy_async(ctx):
    if not is_indexed_schedule(ctx.args):
        return TensorValue(data=ctx.args[0].data, type=ctx.result_type)
    source, index = (arg.data for arg in ctx.args)
    if ctx.op.fill is None:
        data = torch.index_select(source, 0, index)
    else:
        valid = (index >= 0) & (index < source.shape[0])
        data = torch.full(
            (index.numel(), *source.shape[1:]),
            ctx.op.fill,
            dtype=source.dtype,
            device=source.device,
        )
        selected = torch.index_select(source, 0, index[valid])
        positions = torch.nonzero(valid, as_tuple=True)[0]
        data.index_copy_(0, positions, selected)
    return TensorValue(data=data, type=ctx.result_type)


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
    if not is_indexed_copy(call.args):
        if call.target.fill is not None:
            ctx.error(call, "CopyAsync fill requires an index operand")
        return
    index = ctx.type_of(call.args[2])
    for name, operand in (("src", src), ("dst", dst), ("index", index)):
        if not _indexed_layout(operand.layout):
            ctx.error(
                call,
                f"CopyAsync indexed {name} must have a plain or pure Broadcast layout, got {operand.layout}",
            )
    if len(index.shape) != 1 or index.dtype not in (DType.i32, DType.i64):
        ctx.error(call, "CopyAsync index must be rank-1 i32 or i64")
    if not src.shape or tuple(dst.shape) != (index.shape[0], *src.shape[1:]):
        ctx.error(
            call,
            "CopyAsync indexed destination must have shape (index length, *source inner shape)",
        )


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


__all__ = ["ASYNC_WIDTHS", "CopyAsync", "CpAsyncCommit", "CpAsyncWait"]
