"""HIR generic value-form Binary Op (kind-tagged dispatch).

``Binary(BinaryKind.ADD, lhs, rhs)`` is the IR-level form behind the
DSL sugar names (``add`` / ``cmp_eq`` / ``logical_and`` / ...).

HIR Binary is value-form (returns the result Expr) — distinct from
the TIR effect-form ``Binary`` Stmt that writes into ``dst``.
"""

from __future__ import annotations

from tilefoundry.core import ShardTensor
from tilefoundry.evaluator.kernels import binary
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.utils import common_placements
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.pattern import TensorPattern
from tilefoundry.ir.types import DType, ShardLayout, TensorType
from tilefoundry.ir.types.layout import Layout
from tilefoundry.ir.types.shard_layout import (
    shard_layout_of,
)
from tilefoundry.ir.types.utils import check_multilinear_partials
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    broadcast_shapes,
    identity_relations,
    register_access_relation,
    relations_of,
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


@register_op(dialect="tf", category="math", aliases={
    "add": {"kind": BinaryKind.ADD},
    "sub": {"kind": BinaryKind.SUB},
    "mul": {"kind": BinaryKind.MUL},
    "div": {"kind": BinaryKind.DIV},
    "floor_div": {"kind": BinaryKind.FLOOR_DIV},
    "mod": {"kind": BinaryKind.MOD},
    "min": {"kind": BinaryKind.MIN},
    "max": {"kind": BinaryKind.MAX},
    "minimum": {"kind": BinaryKind.MIN},
    "maximum": {"kind": BinaryKind.MAX},
    "cmp_eq": {"kind": BinaryKind.EQ},
    "cmp_ne": {"kind": BinaryKind.NE},
    "cmp_lt": {"kind": BinaryKind.LT},
    "cmp_le": {"kind": BinaryKind.LE},
    "cmp_gt": {"kind": BinaryKind.GT},
    "cmp_ge": {"kind": BinaryKind.GE},
    "logical_and": {"kind": BinaryKind.AND},
    "logical_or": {"kind": BinaryKind.OR},
})
class Binary(Op):
    """Value-form pointwise binary operation."""

    lhs = ParamDef(kind="input", pattern=TensorPattern())
    rhs = ParamDef(kind="input", pattern=TensorPattern())
    kind = ParamDef(kind="attribute", annotation=BinaryKind)


def _merge_layout(a: object, b: object, out_shape: tuple) -> object:
    """Keep equal plain layouts when their shape already matches the result."""
    return a if a == b and a is not None and not isinstance(a, ShardLayout) and tuple(a.shape) == out_shape else None


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

    check_multilinear_partials((("lhs", lhs_ty), ("rhs", rhs_ty)), ctx.target, kind=op.kind)
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
            )
        )
    except ValueError as e:
        ctx.error(call, f"Binary {op.kind.name}: {e}")
    if layout is None:
        layout = Layout(out_shape)
    return TensorType(
        shape=out_shape,
        dtype=out_dtype,
        layout=layout,
        storage=storage,
    )


@register_eval(Binary)
def _eval_binary(ctx):
    a, b = ctx.args[0].tensor, ctx.args[1].tensor
    placements = common_placements(a, b)
    held = (
        binary(ctx.op.kind, a.tensor, b.tensor)
        if not placements
        else tuple((p, binary(ctx.op.kind, a.to_local(p), b.to_local(p))) for p in placements)
    )
    return TensorValue(ShardTensor(ctx.result_type, held))


__all__ = ["Binary"]
