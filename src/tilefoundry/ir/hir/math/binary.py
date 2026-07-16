"""HIR generic value-form Binary Op (kind-tagged dispatch).

``Binary(BinaryKind.ADD, lhs, rhs)`` is the IR-level form behind the
DSL sugar names (``add`` / ``cmp_eq`` / ``logical_and`` / ...).

HIR Binary is value-form (returns the result Expr) — distinct from
the TIR effect-form ``Binary`` Stmt that writes into ``dst``.
"""

from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    build_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.isl_utility import to_domain
from tilefoundry.visitor_registry.relation_build import shape_from_relation
from tilefoundry.visitor_registry.shard_propagate import (
    derive_output_shard_layout,
    partial_reductions_by_axis,
)

from ._helpers import _broadcast, _is_one, _merge_layout, resolve_anchor_storage

_COMPARE_KINDS = {
    BinaryKind.EQ, BinaryKind.NE,
    BinaryKind.LT, BinaryKind.LE,
    BinaryKind.GT, BinaryKind.GE,
}
_LOGICAL_KINDS = {BinaryKind.AND, BinaryKind.OR}
_INT_ONLY_KINDS = {BinaryKind.FLOOR_DIV, BinaryKind.MOD}

@register_op(dialect="tf", category="math")
class Binary(Op):
    """Value-form pointwise binary operation."""
    lhs = ParamDef(kind="input", pattern=Tensor)
    rhs = ParamDef(kind="input", pattern=Tensor)
    kind = ParamDef(kind="attribute", annotation=BinaryKind)


@register_type_relation(Binary)
def _binary_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward access relation for a right-aligned elementwise binary op.

    The iteration domain is the broadcast output shape; the output map is the
    identity. Each operand's map ranges over its own tensor axes (right-aligned
    to the output): axis ``i`` reads iteration dim ``pad + i``, or a constant 0
    when that owned dim is size-1 broadcasting to a larger output dim — so the
    shard engine treats those positions as broadcasts.
    """
    lhs, rhs = input_types
    out_shape = _broadcast(lhs.shape, rhs.shape)
    r = len(out_shape)
    domain, param_map = to_domain(out_shape)
    in_dims = [f"d{i}" for i in range(r)]

    def access(in_shape):
        pad = r - len(in_shape)
        return [
            "0"
            if (_is_one(in_shape[i]) and not _is_one(out_shape[pad + i]))
            else f"d{pad + i}"
            for i in range(len(in_shape))
        ]

    src = "[" + ", ".join(in_dims) + "]"
    dsts = (access(lhs.shape), access(rhs.shape), in_dims)
    maps = tuple(
        isl.map(f"{{ {src} -> [{', '.join(dst)}] }}") for dst in dsts
    )
    return AccessRelationResult(domain=domain, maps=maps, param_map=param_map)


def _check_partial_commutes(call: "Call", ctx: "TypeInferContext", op, la, lb) -> None:
    """Reject a `Partial(reduction)` operand whose reduction the op's math
    does not provably commute with.

    ``ADD``: both-``Partial`` commutes only when both carry ``"sum"`` on the
    same mesh axis (``max``/``min`` self-combine is nonsensical); one
    ``Partial`` against a plain/``Broadcast`` operand commutes for
    ``"max"``/``"min"`` (adding a replicated constant is order-preserving)
    but not ``"sum"`` (today's bug — `sum(x)+b != sum(x+b)`). ``MUL``: one
    ``Partial`` against a plain/``Broadcast`` operand commutes only for
    ``"sum"`` (scaling by a replicated constant distributes over `sum`); both
    ``Partial`` never commutes. Every other kind / shape rejects any
    `Partial` operand — not proven safe.
    """
    lhs_r, rhs_r = partial_reductions_by_axis(la), partial_reductions_by_axis(lb)
    lhs_has = any(r is not None for r in lhs_r)
    rhs_has = any(r is not None for r in rhs_r)
    if not lhs_has and not rhs_has:
        return
    fix = "insert reshard(<arg>, Broadcast) before this consumer"
    if op.kind is BinaryKind.ADD:
        if lhs_has and rhs_has:
            if lhs_r != rhs_r or any(r not in (None, "sum") for r in lhs_r):
                ctx.error(
                    call,
                    f"Binary ADD: Partial(lhs)={tuple(r for r in lhs_r if r is not None)} vs "
                    f"Partial(rhs)={tuple(r for r in rhs_r if r is not None)} is unsound (ADD of two "
                    "Partials only commutes when both are Partial(sum) on "
                    "the same mesh axis) — " + fix,
                )
            return
        arg, r = ("lhs", lhs_r) if lhs_has else ("rhs", rhs_r)
        if any(reduction == "sum" for reduction in r):
            ctx.error(
                call,
                f"Binary ADD: Partial(sum) input on {arg} against a "
                f"plain/Broadcast operand is unsound (sum(x)+b != "
                f"sum(x+b)) — {fix.replace('<arg>', arg)}",
            )
        return
    if op.kind is BinaryKind.MUL:
        if lhs_has and rhs_has:
            ctx.error(
                call,
                f"Binary MUL: Partial(lhs)={tuple(r for r in lhs_r if r is not None)} against "
                f"Partial(rhs)={tuple(r for r in rhs_r if r is not None)} is unsound (MUL of two "
                "Partials is not linear) — " + fix,
            )
        arg, r = ("lhs", lhs_r) if lhs_has else ("rhs", rhs_r)
        if any(reduction not in (None, "sum") for reduction in r):
            ctx.error(
                call,
                f"Binary MUL: Partial({tuple(reduction for reduction in r if reduction is not None)}) input on {arg} against a "
                "plain/Broadcast operand is unsound (the operand's sign is "
                "not statically provable, and a negative scale flips "
                f"max/min) — {fix.replace('<arg>', arg)}",
            )
        return
    arg, r = ("lhs", lhs_r) if lhs_has else ("rhs", rhs_r)
    ctx.error(
        call,
        f"Binary {op.kind.name}: Partial({tuple(reduction for reduction in r if reduction is not None)}) input on {arg} is not "
        f"proven to commute with any reduction — {fix.replace('<arg>', arg)}",
    )


@register_typeinfer(Binary)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    op = call.target
    if not isinstance(op.kind, BinaryKind):
        ctx.error(call, f"Binary: kind must be BinaryKind, got {type(op.kind)}")
    lhs_ty = ctx.type_of(call.args[0])
    rhs_ty = ctx.type_of(call.args[1])
    if lhs_ty.dtype != rhs_ty.dtype:
        ctx.error(call, f"Binary {op.kind.name}: dtype mismatch "
                        f"({lhs_ty.dtype} vs {rhs_ty.dtype})")
    if op.kind in _LOGICAL_KINDS and lhs_ty.dtype != DType.bool:
        ctx.error(call, f"Binary {op.kind.name}: operands must be bool")
    if op.kind in _INT_ONLY_KINDS and lhs_ty.dtype not in (DType.i32, DType.i64):
        ctx.error(call, f"Binary {op.kind.name}: requires integer dtype, "
                        f"got {lhs_ty.dtype}")
    out_dtype = (
        DType.bool
        if op.kind in _COMPARE_KINDS or op.kind in _LOGICAL_KINDS
        else lhs_ty.dtype
    )
    la, lb = lhs_ty.layout, rhs_ty.layout
    _check_partial_commutes(call, ctx, op, la, lb)
    try:
        # Shape and shard share the relation as the single source: the forward
        # relation builds the broadcast domain, the output shape is read back
        # from it, and the shard engine consumes the same maps.
        relation = build_relation(call, (lhs_ty, rhs_ty), ctx)
        out_shape = shape_from_relation(relation)
        shard = None
        if isinstance(la, ShardLayout) or isinstance(lb, ShardLayout):
            shard = derive_output_shard_layout((lhs_ty, rhs_ty), relation, out_shape)
        layout = shard if shard is not None else _merge_layout(la, lb)
    except ValueError as e:
        ctx.error(call, f"Binary {op.kind.name}: {e}")
    return TensorType(
        shape=out_shape,
        dtype=out_dtype,
        layout=layout,
        storage=resolve_anchor_storage(ctx, call, lhs_ty.storage, rhs_ty.storage),
    )


@register_eval(Binary)
def _eval_binary(ctx):

    fns = {
        BinaryKind.ADD: torch.add,
        BinaryKind.SUB: torch.sub,
        BinaryKind.MUL: torch.mul,
        BinaryKind.DIV: torch.div,
        BinaryKind.FLOOR_DIV: torch.floor_divide,
        BinaryKind.MOD: torch.remainder,
        BinaryKind.MIN: torch.minimum,
        BinaryKind.MAX: torch.maximum,
        BinaryKind.EQ: torch.eq,
        BinaryKind.NE: torch.ne,
        BinaryKind.LT: torch.lt,
        BinaryKind.LE: torch.le,
        BinaryKind.GT: torch.gt,
        BinaryKind.GE: torch.ge,
        BinaryKind.AND: torch.logical_and,
        BinaryKind.OR: torch.logical_or,
    }
    out = fns[ctx.op.kind](ctx.args[0].data, ctx.args[1].data)
    return TensorValue(data=out, type=ctx.result_type)


__all__ = ["Binary"]
