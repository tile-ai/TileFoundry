"""Top-K selection HIR primitive.

SGLang baseline kernel K11 (TopK Gating Softmax) emits both the top-k values
and their indices. We model just the selection step here; the upstream Softmax
remains a separate node.

"""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import core as ir_core
from tilefoundry.ir import dim, expr, pattern, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.types import int_tuple


@expr.register_op(dialect='tf', category='tensor')
class TopK(expr.Op):
    """Multi-output (values, indices). Reduces the ``axis`` dim to length ``k``.

    ``largest`` selects the greatest (vs smallest) elements; ``sorted`` requests
    the selected elements be returned in order. Indices are ``i64``.

    ``k`` is a ``ShapeDim`` (``int | DimVar | Expr``): a static ``int``, or a
    dynamic k derived from a context-length ``DimVar`` (e.g.
    ``minimum(512, CTX_LEN // 4)``) is a first-class value propagated through
    typeinfer as a symbolic output-axis length and resolved to a concrete
    ``int`` at evaluation time — not a pad+mask workaround.
    """

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    k = expr.ParamDef(kind="attribute", annotation=types.ShapeDim)
    axis = expr.ParamDef(kind="attribute", annotation=int, default=-1)
    largest = expr.ParamDef(kind="attribute", annotation=bool, default=True)
    sorted = expr.ParamDef(kind="attribute", annotation=bool, default=True)


def static_dim_value(d) -> "int | None":
    """The concrete ``int`` value of a ``ShapeDim`` *d* if statically known, else ``None``.

    The concrete ``int`` value of a ``ShapeDim`` *d* if statically known (a
    raw ``int`` or an int-valued ``Constant``), else ``None`` (a ``DimVar`` or
    a ``Call`` not fully folded — genuinely symbolic).
    """
    if isinstance(d, bool):
        return None
    if isinstance(d, int):
        return d
    if isinstance(d, expr.Constant) and isinstance(d.value, int) and not isinstance(d.value, bool):
        return d.value
    return None


class DimUpperBoundVisitor(ir_core.ExprVisitor[int | None]):
    def visit_Constant(self, d: expr.Constant, ctx=None) -> int | None:
        value = d.value
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def visit_DimVar(self, d: expr.DimVar, ctx=None) -> int:
        return d.hi

    def visit_Call(self, d: expr.Call, ctx=None) -> int | None:
        target = d.target
        if isinstance(target, (expr.DimMin, expr.DimMax, expr.DimAdd, expr.DimMul)):
            a_hi = self.visit(d.args[0], ctx)
            b_hi = self.visit(d.args[1], ctx)
            if a_hi is None or b_hi is None:
                return None
            if isinstance(target, expr.DimMin):
                return min(a_hi, b_hi)
            if isinstance(target, expr.DimMax):
                return max(a_hi, b_hi)
            if isinstance(target, expr.DimAdd):
                return a_hi + b_hi
            return None if a_hi < 0 or b_hi < 0 else a_hi * b_hi
        if isinstance(target, (expr.DimFloorDiv, expr.DimMod)):
            a_hi = self.visit(d.args[0], ctx)
            b_val = static_dim_value(d.args[1])
            if a_hi is None or b_val is None or b_val <= 0:
                return None
            return a_hi // b_val if isinstance(target, expr.DimFloorDiv) else b_val - 1
        return None

    def default_visit(self, d, ctx=None) -> int | None:
        return None


def dim_upper_bound(d) -> "int | None":
    """Return a best-effort inclusive static upper bound.

    Constants are exact and ``DimVar`` uses ``hi``. Supported arithmetic
    recurses when its operands permit a sound bound; subtraction and negative
    multiplication fail open as ``None``.

    See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
    """
    if isinstance(d, bool):
        return None
    if isinstance(d, int):
        return d
    if isinstance(d, expr.Constant):
        v = d.value
        return v if isinstance(v, int) and not isinstance(v, bool) else None
    if isinstance(d, (expr.DimVar, expr.Call)):
        return DimUpperBoundVisitor().visit(d)
    return None


@typeinfer.register(TopK)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TupleType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        ctx.error(call, "x must be at least rank-1")
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        ctx.error(call, f"axis {call.target.axis} out of range for rank {rank}")
    k = call.target.k
    if not dim.is_dimension(k):
        ctx.error(call, f"k must be an int, DimVar, or dim expression, got {type(k).__name__}")
    k_static = static_dim_value(k)
    if k_static is not None and k_static < 0:
        ctx.error(call, f"k must be non-negative, got {k_static}")
    axis_len = x_ty.shape[axis]
    if k_static is not None:
        if isinstance(axis_len, int) and k_static > axis_len:
            ctx.error(call, f"k={k_static} exceeds axis {axis} length {axis_len}")
    elif isinstance(axis_len, int):
        k_hi = dim_upper_bound(k)
        if k_hi is not None and k_hi > axis_len:
            ctx.error(
                call,
                f"k's statically-derivable upper bound {k_hi} exceeds axis "
                f"{axis} length {axis_len}",
            )
    source_shard = types.shard_layout_of(x_ty.layout)
    if source_shard is not None:
        la2ta = int_tuple.mode_axes(int_tuple.compatible_profile(source_shard.layout.shape, x_ty.shape))
        if any(isinstance(a, types.Split) and la2ta[a.axis] == axis for a in source_shard.attrs):
            ctx.error(call, f"selected axis {axis} must not be Split-sharded")

    ir_utils.reject_partials("x", x_ty.layout)
    out_shape = list(x_ty.shape)
    out_shape[axis] = k
    out_shape = tuple(out_shape)

    new_layout = types.Layout(out_shape)
    if source_shard is not None:
        relation = access.relations_of(call, ctx)
        derived = typeinfer.derive_output_shard_layout(
            (x_ty,),
            relation,
            out_shape,
            fresh_strides=True,
            ctx=ctx,
            storage=x_ty.storage,
        )

        new_layout = (
            derived
            if derived is not None
            else types.ShardLayout.from_logical(
                out_shape,
                source_shard.mesh,
                source_shard.attrs,
                storage=x_ty.storage,
                owners=ctx.target.storage_owners
                if ctx is not None and ctx.target is not None
                else expr.StorageOwners((), {}),
            )
        )
    values_ty = types.TensorType(
        shape=out_shape, dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage
    )
    indices_ty = types.TensorType(
        shape=out_shape, dtype=expr.DType.i64, layout=new_layout, storage=x_ty.storage
    )
    return types.TupleType(fields=(values_ty, indices_ty))


@access.register_relation(TopK)
def topk_access_relation(call: "expr.Call", ctx: "TypeInferContext") -> tuple[access.Relation, ...]:
    """One selection per kept position, scanning the whole axis it selects from.

    The axis a selection lands on is not a view of the axis it came from -- which
    element is picked is a value's answer -- so the space walked names both: the
    ``k`` positions produced and the coordinate scanned to produce them. The
    result projects the first; the source reads the second.
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    picked = static_dim_value(call.target.k)
    if picked is None:
        raise NotImplementedError(
            f"TopK access relation: k must be a static extent here, got {call.target.k!r}"
        )
    dims = [f"d{index}" for index in range(rank + 1)]
    walked = ", ".join(dims)
    scanned = dims[rank]
    reads = ", ".join(scanned if index == axis else dims[index] for index in range(rank))
    kept = ", ".join(dims[:rank])
    written = access.Relation(isl.map(f"{{ [{walked}] -> [{kept}] }}"))
    out_shape = (*x_ty.shape[:axis], picked, *x_ty.shape[axis + 1 :])
    return access.iterating(
        (*out_shape, x_ty.shape[axis]),
        (access.Relation(isl.map(f"{{ [{walked}] -> [{reads}] }}")), written, written),
    )


def local_dim_bindings(type, data) -> dict:
    """Recover bare ``DimVar`` bindings by pairing *type*'s shape with *data*'s.

    This call-local evaluator cannot recover dimensions that occur only in
    another enclosing function argument.
    """
    bindings: dict = {}
    for axis, dim_value in enumerate(type.shape):
        if isinstance(dim_value, expr.DimVar) and axis < len(data.shape):
            bindings[dim_value.name] = int(data.shape[axis])
    return bindings


@evaluator.register(TopK)
def eval_topk(ctx):
    fields = ctx.result_type.fields

    def topk(x, type):
        k = evaluator.resolve_dim(ctx.op.k, local_dim_bindings(type, x))
        vals, idx = utils.torch.topk(x, k, dim=ctx.op.axis, largest=ctx.op.largest, sorted=ctx.op.sorted)
        return tuple(
            out.to(evaluator.to_torch_dtype(field.dtype)) for out, field in zip((vals, idx), fields, strict=True)
        )

    x = ctx.args[0].tensor
    if x.is_global:
        held = topk(x.tensor, x.type)
    else:
        piece = core.local_type(x.type)
        each = [(p, topk(t, piece)) for p, t in x.tensor]
        held = tuple(tuple((p, out[i]) for p, out in each) for i in range(len(fields)))
    return evaluator.TupleValue(
        tuple(evaluator.TensorValue(core.ShardTensor(field, data)) for field, data in zip(fields, held, strict=True))
    )


__all__ = ["TopK"]


@cost.register(TopK)
def topk_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    inputs = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    source = inputs[0]
    dtype = source.dtype if isinstance(source, types.TensorType) else expr.DType.f32
    return cost.operation(dtype, ir_utils.numel(source), cost.traffic(inputs, output))
