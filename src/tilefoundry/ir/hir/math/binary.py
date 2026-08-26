"""HIR generic value-form Binary Op (kind-tagged dispatch).

``Binary(BinaryKind.ADD, lhs, rhs)`` is the IR-level form behind the
DSL sugar names (``add`` / ``cmp_eq`` / ``logical_and`` / ...).

HIR Binary is value-form (returns the result Expr) — distinct from
the TIR effect-form ``Binary`` Stmt that writes into ``dst``.
"""

from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import broadcast_shapes, resolve_anchor_storage
from tilefoundry.ir.hir._shard_checks import check_multilinear_partials
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, canonical_shard_layout, try_c_order_strides
from tilefoundry.ir.types.shard.shard_layout import Broadcast, ShardLayout, shard_layout_of
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    BoundaryRelation,
    broadcast_access,
    coordinates_of,
    identity_access,
    iterating,
    register_access_relation,
    shape_from_relation,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout

_COMPARE_KINDS = {
    BinaryKind.EQ,
    BinaryKind.NE,
    BinaryKind.LT,
    BinaryKind.LE,
    BinaryKind.GT,
    BinaryKind.GE,
}
_LOGICAL_KINDS = {BinaryKind.AND, BinaryKind.OR}
_INT_ONLY_KINDS = {BinaryKind.FLOOR_DIV, BinaryKind.MOD}


@register_op
class Binary(Op):
    """Value-form pointwise binary operation."""

    lhs = ParamDef(kind="input", pattern=Tensor)
    rhs = ParamDef(kind="input", pattern=Tensor)
    kind = ParamDef(kind="attribute", annotation=BinaryKind)


def _merge_layout(a: object, b: object, out_shape: tuple) -> object:
    """Merge two non-sharded operand layouts.

    Merge two non-sharded operand layouts. Equal layouts or one ``None``
    pass through. Two fully-replicated (all-``Broadcast``) ``ShardLayout``s are
    mesh-agnostic (the data is replicated everywhere) so the first is kept.
    Any other genuine mismatch raises — there is no silent lhs pick; a real
    shard mismatch is propagated through the shard engine, not merged here.
    """
    if a == b:
        if a is None:
            return None
        layout = a
    elif a is None:
        layout = b
    elif b is None:
        layout = a
    elif not isinstance(a, ShardLayout) and not isinstance(b, ShardLayout):
        return None
    else:
        layout = None
    if layout is not None:
        if isinstance(layout, ShardLayout):
            if all(isinstance(attr, Broadcast) for attr in layout.attrs):
                return canonical_shard_layout(out_shape, layout.mesh, layout.attrs)
            return layout
        if tuple(layout.shape) == out_shape:
            return layout
        return None
    replicated = [
        layout
        for layout in (a, b)
        if isinstance(layout, ShardLayout)
        and all(isinstance(attr, Broadcast) for attr in layout.attrs)
    ]
    if replicated and all(
        not isinstance(layout, ShardLayout) or layout in replicated for layout in (a, b)
    ):
        first = replicated[0]
        return canonical_shard_layout(out_shape, first.mesh, first.attrs)
    raise ValueError(f"incompatible operand layouts {a!r} vs {b!r}")


@register_access_relation(Binary)
def _elementwise_binary(call: "Call", ctx) -> AccessRelations:
    shapes = tuple(tuple(ctx.type_of(arg).shape) for arg in call.args)
    out_shape = broadcast_shapes(*shapes)
    produced = 1
    for extent in out_shape:
        produced *= extent if isinstance(extent, int) else 1
    return iterating(
        out_shape,
        AccessRelations(
            inputs=tuple(
                BoundaryRelation(broadcast_access(out_shape, shape))
                for arg, shape in zip(call.args, shapes)
            ),
            outputs=(BoundaryRelation(identity_access(len(out_shape))),),
        ),
    )


@register_typeinfer(Binary)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    op = call.target
    if not isinstance(op.kind, BinaryKind):
        ctx.error(call, f"Binary: kind must be BinaryKind, got {type(op.kind)}")
    lhs_ty = ctx.type_of(call.args[0])
    rhs_ty = ctx.type_of(call.args[1])
    if lhs_ty.dtype != rhs_ty.dtype:
        ctx.error(
            call,
            f"Binary {op.kind.name}: dtype mismatch "
            f"({lhs_ty.dtype.name} vs {rhs_ty.dtype.name}); operands are never "
            f"promoted, and a Python literal is f32 or i64 like any other. "
            f"Write the dtype you want: "
            f"tf.cast(<operand>, dtype={lhs_ty.dtype.name!r}). "
            f"See `tilefoundry spec dsl binary`",
        )
    if op.kind in _LOGICAL_KINDS and lhs_ty.dtype != DType.bool:
        ctx.error(call, f"Binary {op.kind.name}: operands must be bool")
    if op.kind in _INT_ONLY_KINDS and lhs_ty.dtype not in (DType.i32, DType.i64):
        ctx.error(call, f"Binary {op.kind.name}: requires integer dtype, got {lhs_ty.dtype}")
    out_dtype = (
        DType.bool if op.kind in _COMPARE_KINDS or op.kind in _LOGICAL_KINDS else lhs_ty.dtype
    )
    la, lb = lhs_ty.layout, rhs_ty.layout

    if op.kind is BinaryKind.ADD:
        allowed_reduction, commutes_jointly = frozenset({"max", "min"}), frozenset({"sum"})
    elif op.kind is BinaryKind.MUL:
        allowed_reduction, commutes_jointly = frozenset({"sum"}), frozenset()
    else:
        allowed_reduction, commutes_jointly = frozenset(), frozenset()
    check_multilinear_partials(
        ctx,
        call,
        (("lhs", lhs_ty), ("rhs", rhs_ty)),
        allowed_reduction=allowed_reduction,
        commutes_jointly=commutes_jointly,
    )
    try:
        relation = coordinates_of(call, ctx)
        out_shape = shape_from_relation(
            relation, broadcast_shapes(lhs_ty.shape, rhs_ty.shape)
        )
        shard = None
        if shard_layout_of(la) is not None or shard_layout_of(lb) is not None:
            shard = derive_output_shard_layout((lhs_ty, rhs_ty), relation, out_shape)
        layout = (
            shard
            if shard is not None
            else _merge_layout(
                shard_layout_of(la) or la,
                shard_layout_of(lb) or lb,
                out_shape,
            )
        )
    except ValueError as e:
        ctx.error(call, f"Binary {op.kind.name}: {e}")
    storage = resolve_anchor_storage(ctx, call, lhs_ty.storage, rhs_ty.storage)
    if layout is None and storage in (StorageKind.RMEM, StorageKind.SMEM) and out_shape:
        layout = Layout(shape=out_shape, strides=try_c_order_strides(out_shape))
    return TensorType(
        shape=out_shape,
        dtype=out_dtype,
        layout=layout,
        storage=storage,
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
