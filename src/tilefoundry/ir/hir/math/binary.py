"""HIR generic value-form Binary Op (kind-tagged dispatch).

``Binary(BinaryKind.ADD, lhs, rhs)`` is the IR-level form behind the
DSL sugar names (``add`` / ``cmp_eq`` / ``logical_and`` / ...).

HIR Binary is value-form (returns the result Expr) — distinct from
the TIR effect-form ``Binary`` Stmt that writes into ``dst``.
"""

from __future__ import annotations

from tilefoundry.evaluator.kernels import binary
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.hir._shard_checks import check_multilinear_partials
from tilefoundry.ir.pattern import TensorPattern
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.layout import default_layout, resolve_layout
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    ShardLayout,
    shard_layout_of,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    broadcast_shapes,
    identity_relations,
    register_access_relation,
    relations_of,
    shape_from_relation,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout
from tilefoundry.visitor_registry.type_defaults import output_shard_layout

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

    lhs = ParamDef(kind="input", pattern=TensorPattern())
    rhs = ParamDef(kind="input", pattern=TensorPattern())
    kind = ParamDef(kind="attribute", annotation=BinaryKind)


def _merge_layout(a: object, b: object, out_shape: tuple, *, ctx=None, storage=None) -> object:
    """Merge two non-sharded operand layouts.

    Merge two non-sharded operand layouts. Equal layouts or one ``None``
    pass through. Two fully-replicated (all-``Broadcast``) ``ShardLayout``s are
    mesh-agnostic (the data is replicated everywhere) so the first is kept.
    Any other genuine mismatch raises — there is no silent lhs pick; a real
    shard mismatch is propagated through the shard engine, not merged here.
    """
    if resolve_layout(a, out_shape) == resolve_layout(b, out_shape):
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
                return output_shard_layout(
                    out_shape, layout.mesh, layout.attrs, ctx=ctx, storage=storage
                )
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
        return output_shard_layout(out_shape, first.mesh, first.attrs, ctx=ctx, storage=storage)
    raise ValueError(f"incompatible operand layouts {a!r} vs {b!r}")


register_access_relation(Binary)(identity_relations)


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
    storage = resolve_anchor_storage(ctx, call, lhs_ty.storage, rhs_ty.storage)
    try:
        relation = relations_of(call, ctx)
        out_shape = shape_from_relation(
            relation[len(call.args)], broadcast_shapes(lhs_ty.shape, rhs_ty.shape)
        )
        shard = None
        if shard_layout_of(la) is not None or shard_layout_of(lb) is not None:
            shard = derive_output_shard_layout(
                (lhs_ty, rhs_ty), relation, out_shape, ctx=ctx, storage=storage
            )
        layout = (
            shard
            if shard is not None
            else _merge_layout(
                shard_layout_of(la) or la,
                shard_layout_of(lb) or lb,
                out_shape,
                ctx=ctx,
                storage=storage,
            )
        )
    except ValueError as e:
        ctx.error(call, f"Binary {op.kind.name}: {e}")
    if layout is None:
        layout = default_layout(out_shape)
    return TensorType(
        shape=out_shape,
        dtype=out_dtype,
        layout=layout,
        storage=storage,
    )


@register_eval(Binary)
def _eval_binary(ctx):
    out = binary(ctx.op.kind, ctx.args[0].data, ctx.args[1].data)
    return TensorValue(data=out, type=ctx.result_type)


__all__ = ["Binary"]
