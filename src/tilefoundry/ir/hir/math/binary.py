"""HIR generic value-form Binary Op (kind-tagged dispatch).

``Binary(BinaryKind.ADD, lhs, rhs)`` is the IR-level form behind the
DSL sugar names (``add`` / ``cmp_eq`` / ``logical_and`` / ...).

HIR Binary is value-form (returns the result Expr) — distinct from
the TIR effect-form ``Binary`` Stmt that writes into ``dst``.
"""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils

COMPARE_KINDS = frozenset(
    {
        expr.BinaryKind.EQ,
        expr.BinaryKind.NE,
        expr.BinaryKind.LT,
        expr.BinaryKind.LE,
        expr.BinaryKind.GT,
        expr.BinaryKind.GE,
    }
)
LOGICAL_KINDS = frozenset({expr.BinaryKind.AND, expr.BinaryKind.OR})
INT_ONLY_KINDS = frozenset({expr.BinaryKind.FLOOR_DIV, expr.BinaryKind.MOD})


@expr.register_op(dialect="tf", category="math", aliases={
    "add": {"kind": expr.BinaryKind.ADD},
    "sub": {"kind": expr.BinaryKind.SUB},
    "mul": {"kind": expr.BinaryKind.MUL},
    "div": {"kind": expr.BinaryKind.DIV},
    "floor_div": {"kind": expr.BinaryKind.FLOOR_DIV},
    "mod": {"kind": expr.BinaryKind.MOD},
    "min": {"kind": expr.BinaryKind.MIN},
    "max": {"kind": expr.BinaryKind.MAX},
    "minimum": {"kind": expr.BinaryKind.MIN},
    "maximum": {"kind": expr.BinaryKind.MAX},
    "cmp_eq": {"kind": expr.BinaryKind.EQ},
    "cmp_ne": {"kind": expr.BinaryKind.NE},
    "cmp_lt": {"kind": expr.BinaryKind.LT},
    "cmp_le": {"kind": expr.BinaryKind.LE},
    "cmp_gt": {"kind": expr.BinaryKind.GT},
    "cmp_ge": {"kind": expr.BinaryKind.GE},
    "logical_and": {"kind": expr.BinaryKind.AND},
    "logical_or": {"kind": expr.BinaryKind.OR},
})
class Binary(expr.Op):
    """Value-form pointwise binary operation."""

    lhs = expr.ParamDef(kind="input", pattern=pattern.Tensor())
    rhs = expr.ParamDef(kind="input", pattern=pattern.Tensor())
    kind = expr.ParamDef(kind="attribute", annotation=expr.BinaryKind)


def merge_layout(a: object, b: object, out_shape: tuple) -> object:
    """Keep equal plain layouts when their shape already matches the result."""
    if (
        a == b
        and a is not None
        and not isinstance(a, types.ShardLayout)
        and tuple(a.shape) == out_shape
    ):
        return a
    return None


access.register_relation(Binary)(access.identity_relations)


@typeinfer.register(Binary)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    op = call.target
    if not isinstance(op.kind, expr.BinaryKind):
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
    if op.kind in LOGICAL_KINDS and lhs_ty.dtype != expr.DType.bool:
        ctx.error(call, f"Binary {op.kind.name}: operands must be bool")
    if op.kind in INT_ONLY_KINDS and lhs_ty.dtype not in (expr.DType.i32, expr.DType.i64):
        ctx.error(call, f"Binary {op.kind.name}: requires integer dtype, got {lhs_ty.dtype}")
    out_dtype = (
        expr.DType.bool if op.kind in COMPARE_KINDS or op.kind in LOGICAL_KINDS else lhs_ty.dtype
    )
    la, lb = lhs_ty.layout, rhs_ty.layout

    ir_utils.check_multilinear_partials((("lhs", lhs_ty), ("rhs", rhs_ty)), ctx.target, kind=op.kind)
    storage = typeinfer.resolve_anchor_storage(ctx, call, lhs_ty.storage, rhs_ty.storage)
    try:
        relation = access.relations_of(call, ctx)
        out_shape = access.shape_from_relation(
            relation[len(call.args)], access.broadcast_shapes(lhs_ty.shape, rhs_ty.shape)
        )
        shard = None
        if types.shard_layout_of(la) is not None or types.shard_layout_of(lb) is not None:
            shard = typeinfer.derive_output_shard_layout(
                (lhs_ty, rhs_ty), relation, out_shape, ctx=ctx, storage=storage
            )
        layout = (
            shard
            if shard is not None
            else merge_layout(
                types.shard_layout_of(la) or la,
                types.shard_layout_of(lb) or lb,
                out_shape,
            )
        )
    except ValueError as e:
        ctx.error(call, f"Binary {op.kind.name}: {e}")
    if layout is None:
        layout = types.Layout(out_shape)
    return types.TensorType(
        shape=out_shape,
        dtype=out_dtype,
        layout=layout,
        storage=storage,
    )


@evaluator.register(Binary)
def eval_binary(ctx):
    a, b = ctx.args[0].tensor, ctx.args[1].tensor
    placements = evaluator.common_placements(a, b)
    held = (
        evaluator.binary(ctx.op.kind, a.tensor, b.tensor)
        if not placements
        else tuple((p, evaluator.binary(ctx.op.kind, a.to_local(p), b.to_local(p))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["Binary", "merge_layout"]


PREDICATES = frozenset(
    {
        expr.BinaryKind.EQ,
        expr.BinaryKind.NE,
        expr.BinaryKind.LT,
        expr.BinaryKind.LE,
        expr.BinaryKind.GT,
        expr.BinaryKind.GE,
        expr.BinaryKind.AND,
        expr.BinaryKind.OR,
    }
)


@cost.register(Binary)
def binary_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """Arithmetic is floating-point work; comparing and combining truths is not.

    A comparison over floats produces booleans, and calling that a bool FLOP
    asks the target for a rate no machine publishes. What it really asks for is
    a predicate, which is a service a machine does state a throughput for.
    ``_PREDICATES`` holds the kinds whose result is one, whatever the operands
    were.
    """
    kind = call.target.kind
    if kind in PREDICATES:
        return cost.serviced(call, ctx, "predicate")
    return cost.elementwise(call, ctx)
