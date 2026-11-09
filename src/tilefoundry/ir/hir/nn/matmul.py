"""HIR batched ``MatMul`` Op."""

from __future__ import annotations

import math
from typing import Literal

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect='tf', category='nn')
class MatMul(expr.Op):
    """Batched matrix multiplication with explicit physical matrix-axis order."""

    lhs = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    rhs = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    a_layout = expr.ParamDef(kind="attribute", annotation=Literal["MK", "KM"], default="MK")
    b_layout = expr.ParamDef(kind="attribute", annotation=Literal["NK", "KN"], default="KN")
    out_dtype = expr.ParamDef(kind="attribute", annotation=expr.DType, optional=True, default=None)

    def __init__(self, **attrs) -> None:
        out_dtype = attrs.get("out_dtype")
        if isinstance(out_dtype, str):
            attrs["out_dtype"] = expr.DType.from_name(out_dtype)
        super().__init__(**attrs)


def matmul_axes(op: MatMul) -> tuple[int, int, int, int]:
    """Return physical ``(A.M, A.K, B.N, B.K)`` axes for the layout literals."""
    if op.a_layout == "MK":
        a_m, a_k = -2, -1
    elif op.a_layout == "KM":
        a_m, a_k = -1, -2
    else:
        raise ValueError(f"MatMul: a_layout must be 'MK' or 'KM', got {op.a_layout!r}")
    if op.b_layout == "NK":
        b_n, b_k = -2, -1
    elif op.b_layout == "KN":
        b_n, b_k = -1, -2
    else:
        raise ValueError(f"MatMul: b_layout must be 'NK' or 'KN', got {op.b_layout!r}")
    return a_m, a_k, b_n, b_k


def k_split_axes(t, k_tensor_axis: int) -> "frozenset[int]":
    """The mesh axes on which *t* splits its contraction (K) tensor axis."""
    layout = types.shard_layout_of(t.layout)
    if layout is None:
        return frozenset()
    targets = types.split_target_axes(layout, t.shape)
    return frozenset(p for p, ax in enumerate(targets) if ax == k_tensor_axis)


@access.register_relation(MatMul)
def matmul_access_relation(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """Every coordinate of each operand a contraction reaches, read once."""
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    return access.matmul_relations(lhs.shape, rhs.shape, matmul_axes(call.target))


def elements(shape: tuple) -> int:
    """How many elements a shape of numbers holds."""
    counted = 1
    for extent in shape:
        counted *= extent if isinstance(extent, int) else 1
    return counted


def matmul_result_dtype(op: MatMul, lhs: types.TensorType) -> expr.DType:
    """The result dtype: ``out_dtype`` when stated, else the operand dtype."""
    return lhs.dtype if op.out_dtype is None else op.out_dtype


def matmul_result_shape_and_layout(call: "expr.Call", ctx: "TypeInferContext") -> tuple:
    """Check the operands against each other and derive the result shape and layout.

    Every target's MatMul rule reads its shape and layout here; only the result
    storage is the target's to decide.
    """
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    try:
        a_m, a_k, b_n, b_k = matmul_axes(call.target)
    except ValueError as error:
        ctx.error(call, str(error).removeprefix("MatMul: "))
    if lhs.dtype != rhs.dtype:
        ctx.error(call, f"MatMul dtype mismatch: {lhs.dtype} vs {rhs.dtype}")
    if len(lhs.shape) < 2 or len(rhs.shape) < 2:
        ctx.error(call, "MatMul requires rank >= 2 on both operands")
    if access.broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2], raising=False) is None:
        ctx.error(call, f"MatMul batch-dim mismatch {lhs.shape[:-2]} vs {rhs.shape[:-2]}")
    if lhs.shape[a_k] != rhs.shape[b_k]:
        ctx.error(
            call,
            f"MatMul contraction-dim mismatch: lhs K={lhs.shape[a_k]} vs rhs K={rhs.shape[b_k]}",
        )

    if k_split_axes(lhs, a_k % len(lhs.shape)) != k_split_axes(rhs, b_k % len(rhs.shape)):
        ctx.error(
            call,
            "MatMul contraction dim K must be split on the same mesh axes for both operands",
        )

    ir_utils.check_multilinear_partials(
        (("lhs", lhs), ("rhs", rhs)), ctx.target, kind=expr.BinaryKind.MUL
    )

    relation = access.relations_of(call, ctx)

    out_batch = access.broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2], raising=False)
    out_shape = access.shape_from_relation(
        relation[len(call.args)], (*out_batch, lhs.shape[a_m], rhs.shape[b_n], lhs.shape[a_k])
    )
    k_domain_dim = len(out_shape)
    try:
        shard = typeinfer.derive_output_shard_layout(
            (lhs, rhs),
            relation,
            out_shape,
            partial_reduction_dims=frozenset({k_domain_dim}),
            ctx=ctx,
            storage=lhs.storage,
        )
    except ValueError as e:
        ctx.error(call, str(e))
    layout = shard
    if layout is None:
        held = types.shard_layout_of(lhs.layout) or lhs.layout
        if isinstance(held, types.ShardLayout):
            layout = types.ShardLayout.from_logical(
                out_shape,
                held.mesh,
                held.attrs,
                storage=lhs.storage,
                owners=ctx.target.storage_owners
                if ctx is not None and ctx.target is not None
                else expr.StorageOwners((), {}),
            )
        elif tuple(held.shape) == tuple(out_shape):
            layout = held
        else:
            layout = types.Layout(out_shape)
    return out_shape, layout


@typeinfer.register(MatMul)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    shape, layout = matmul_result_shape_and_layout(call, ctx)
    on_chip = {lhs.storage, rhs.storage} & {expr.StorageKind.SMEM, expr.StorageKind.RMEM}
    if on_chip and ctx.resolve_target() is None:
        storage = expr.StorageKind.UMAT
    else:
        storage = typeinfer.resolve_anchor_storage(ctx, call, lhs.storage, rhs.storage)
    return types.TensorType(
        shape=shape, dtype=matmul_result_dtype(call.target, lhs), layout=layout, storage=storage
    )


@evaluator.register(MatMul)
def eval_matmul(ctx):
    """Multiply in the operand dtype, or widen to f32 and round once.

    The operand-dtype product serves when the result keeps that dtype and the
    operand is not an 8-bit float, which torch cannot multiply.
    """
    operand = ctx.args[0].type.dtype
    fp8 = isinstance(operand, expr.FloatDType) and operand.bit_width == 8

    def matmul(lhs, rhs):
        if ctx.op.a_layout == "KM":
            lhs = lhs.transpose(-1, -2)
        if ctx.op.b_layout == "NK":
            rhs = rhs.transpose(-1, -2)
        if ctx.result_type.dtype == operand and not fp8:
            return utils.torch.matmul(lhs, rhs)
        return utils.torch.matmul(lhs.float(), rhs.float()).to(evaluator.to_torch_dtype(ctx.result_type.dtype))

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        matmul(*(x.tensor for x in xs))
        if not placements
        else tuple((p, matmul(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@cost.register(MatMul)
def matmul_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """One multiply and one add per multiply-accumulate: 2 * batch * m * k * n."""
    lhs, rhs = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    if not all(isinstance(type, types.TensorType) for type in (lhs, rhs, output)):
        raise ValueError("MatMul cost requires tensor inputs and output")
    logical_lhs = ctx.type_of(call.args[0])
    if not isinstance(logical_lhs, types.TensorType):
        raise ValueError("MatMul cost requires a tensor lhs")
    _, a_k, _, _ = matmul_axes(call.target)
    k_axis = a_k % len(logical_lhs.shape)
    k = math.prod(
        extent
        for extent, logical_axis in zip(
            lhs.shape, access.logical_axes_of(lhs, logical_lhs)
        )
        if logical_axis == k_axis
    )
    flops = 2 * ir_utils.numel(output) * k
    return cost.Cost({lhs.dtype: flops}, cost.traffic((lhs, rhs), output))


__all__ = ["MatMul", "matmul_result_dtype", "matmul_result_shape_and_layout"]
