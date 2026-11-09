"""Effect-form TIR Ops for asynchronous (``cp.async``) gmem→smem staging."""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, evaluator, typeinfer, verify
from tilefoundry.ir import expr, pattern, types

ASYNC_WIDTHS = (4, 8, 16)


def is_indexed_copy(args) -> bool:
    """A TIR CopyAsync supplies src, dst, and the optional index input."""
    return len(args) == 3


class CopyModes(pattern.SameModesConstraint):
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
            pattern.WholeVectors(pattern.Wildcard("width"), "dtype", ASYNC_WIDTHS).available_widths(
                type_.layout, {"dtype": type_.dtype}
            )
        )
        for type_ in (source, destination)
    )
    shared = widths[0] & widths[1]
    if not shared or not CopyModes("src", "dst").holds(
        {"src": source, "dst": destination, "index": True}
    ):
        raise ValueError("indexed CopyAsync requires common aligned vectors inside a row")
    return max(shared)


def indexed_layout(layout) -> bool:
    """Every issuer must see the complete table, destination, and index."""
    if isinstance(layout, types.Layout):
        return True
    if isinstance(layout, types.ShardLayout):
        return not any(
            isinstance(attr, (types.Split, types.Partial)) for attr in layout.attrs
        ) and indexed_layout(layout.layout)
    if isinstance(layout, types.ComposedLayout):
        return (layout.outer is None or indexed_layout(layout.outer)) and (
            layout.inner is None or indexed_layout(layout.inner)
        )
    return isinstance(layout, types.Swizzle)


@expr.register_op(dialect="T", category="async", name="copy_async")
class CopyAsync(expr.Op):
    """Async gmem→smem copy (``cp.async.cg.shared.global``); non-blocking."""

    capability = expr.OpCapability("cp.async")
    execution_mesh = pattern.thread_execution_mesh()

    src = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        pattern=pattern.operand_tile(
            0,
            expr.StorageKind.GMEM,
            pattern.whole_vectors(0, ASYNC_WIDTHS),
            execution_mesh,
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.WRITE,
        pattern=pattern.operand_tile(
            1,
            expr.StorageKind.SMEM,
            pattern.whole_vectors(1, ASYNC_WIDTHS),
            execution_mesh,
        ),
    )
    between = (
        pattern.SameConstraint("dtype", "src", "dst"),
        pattern.DistinctConstraint("storage", "src", "dst"),
        CopyModes("src", "dst"),
    )
    index = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        optional=True,
        default=None,
        pattern=pattern.Tensor(shape=(None,), dtype=pattern.Or(expr.DType.i32, expr.DType.i64)),
    )
    fill = expr.ParamDef(kind="attribute", annotation=float | None, default=None)


@typeinfer.register(CopyAsync)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


@access.register_relation(CopyAsync)
def copy_async_access(call, ctx):
    if not is_indexed_copy(call.args):
        return access.identity_relations(call, ctx)
    source = ctx.type_of(call.args[0])
    index = ctx.type_of(call.args[2])
    source_access, index_access, out = access.gather_relations(source, index, 0)
    return source_access, out, index_access, out


@evaluator.register(CopyAsync)
def eval_scheduled_copy_async(ctx):
    if not is_indexed_copy(ctx.args):
        return evaluator.TensorValue(core.ShardTensor(ctx.result_type, ctx.args[0].data))
    source, index = ctx.args[0].data, ctx.args[2].data
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, evaluator.gather(source, index, 0, ctx.op.fill)))


@verify.register(CopyAsync)
def verify_copy_async(call: "Call", ctx: "verify.Context") -> None:
    """Check indexed-copy geometry and the index-dependent fill option."""
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if not is_indexed_copy(call.args):
        if call.target.fill is not None:
            ctx.error(call, "CopyAsync fill requires an index operand")
        return
    index = ctx.type_of(call.args[2])
    for name, operand in (("src", src), ("dst", dst), ("index", index)):
        if not indexed_layout(operand.layout):
            ctx.error(
                call,
                f"CopyAsync indexed {name} must have a plain or pure Broadcast layout, got {operand.layout}",
            )
    if len(index.shape) != 1 or index.dtype not in (expr.DType.i32, expr.DType.i64):
        ctx.error(call, "CopyAsync index must be rank-1 i32 or i64")
    if not src.shape or tuple(dst.shape) != (index.shape[0], *src.shape[1:]):
        ctx.error(
            call,
            "CopyAsync indexed destination must have shape (index length, *source inner shape)",
        )


@expr.register_op(dialect="T", category="async", name="cp_async_commit")
class CpAsyncCommit(expr.Op):
    """Close the current in-flight ``cp.async`` group (``commit_group``)."""

    execution_mesh = pattern.thread_execution_mesh()


@typeinfer.register(CpAsyncCommit)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


@expr.register_op(dialect="T", category="async", name="cp_async_wait")
class CpAsyncWait(expr.Op):
    """Wait until all but the ``n`` newest committed async groups arrive."""

    execution_mesh = pattern.thread_execution_mesh()

    n = expr.ParamDef(
        kind="attribute",
        annotation=int,
        pattern=pattern.Wildcard("wait_groups", predicates=(pattern.Wildcard("wait_groups") >= 0,)),
        default=0,
    )


@typeinfer.register(CpAsyncWait)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


__all__ = [
    "ASYNC_WIDTHS",
    "CopyAsync",
    "CpAsyncCommit",
    "CpAsyncWait",
    "indexed_width",
    "is_indexed_copy",
]


access.register_relation(CpAsyncCommit)(access.identity_relations)
access.register_relation(CpAsyncWait)(access.identity_relations)


@verify.register(CpAsyncCommit)
def verify_cpasynccommit(call, ctx) -> None:
    """Admit a control effect with no input contract.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    return None


@verify.register(CpAsyncWait)
def verify_cpasyncwait(call, ctx) -> None:
    """Admit a control effect with no input contract.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    return None
