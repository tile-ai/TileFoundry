"""Top-K selection HIR primitive.

SGLang baseline kernel K11 (TopK Gating Softmax) emits both the top-k values
and their indices. We model just the selection step here; the upstream Softmax
remains a separate node.

"""
from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue, to_torch_dtype
from tilefoundry.ir.core import Call, Constant, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimVar,
    is_dim_expr,
)
from tilefoundry.ir.types.shape_dim import ShapeDim
from tilefoundry.ir.types.shard import Layout, try_c_order_strides
from tilefoundry.ir.types.shard.shard_layout import (
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    AccessRelations,
    build_relation,
    register_access_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class TopK(Op):
    """Multi-output (values, indices). Reduces the ``axis`` dim to length ``k``.

    ``largest`` selects the greatest (vs smallest) elements; ``sorted`` requests
    the selected elements be returned in order. Indices are ``i64``.

    ``k`` is a ``ShapeDim`` (``int | DimVar | Expr``): a static ``int``, or a
    dynamic k derived from a context-length ``DimVar`` (e.g.
    ``dim_min(512, CTX_LEN // 4)``) is a first-class value propagated through
    typeinfer as a symbolic output-axis length and resolved to a concrete
    ``int`` at evaluation time — not a pad+mask workaround.
    """
    x = ParamDef(kind="input", pattern=Tensor)
    k = ParamDef(kind="attribute", annotation=ShapeDim)
    axis = ParamDef(kind="attribute", annotation=int, default=-1)
    largest = ParamDef(kind="attribute", annotation=bool, default=True)
    sorted = ParamDef(kind="attribute", annotation=bool, default=True)


def _static_dim_value(d) -> "int | None":
    """The concrete ``int`` value of a ``ShapeDim`` *d* if statically known (a
    raw ``int`` or an int-valued ``Constant``), else ``None`` (a ``DimVar`` or
    a ``Call`` not fully folded — genuinely symbolic)."""
    if isinstance(d, bool):
        return None
    if isinstance(d, int):
        return d
    if isinstance(d, Constant) and isinstance(d.value, int) and not isinstance(d.value, bool):
        return d.value
    return None


def _dim_upper_bound(d) -> "int | None":
    """Best-effort static upper bound (inclusive) for a ``ShapeDim``; ``None``
    when not statically derivable (fails open — the caller only flags an
    oversized ``k`` when this resolves AND the axis length is static).

    Covers ``int``/``Constant`` (exact), ``DimVar`` (``hi - 1``), ``DimMin``/
    ``DimMax``/``DimAdd``/``DimMul`` (recurse both operands; ``DimMul`` gives
    up on a negative operand bound), and ``DimFloorDiv``/``DimMod`` (recurse
    the dividend; the divisor must itself be a static positive int).
    ``DimSub`` is not covered — bounding ``a - b`` needs ``b``'s lower bound,
    which this helper has no counterpart for.
    """
    if isinstance(d, bool):
        return None
    if isinstance(d, int):
        return d
    if isinstance(d, Constant):
        v = d.value
        return v if isinstance(v, int) and not isinstance(v, bool) else None
    if isinstance(d, DimVar):
        return d.hi - 1
    if isinstance(d, Call):
        target = d.target
        if isinstance(target, (DimMin, DimMax, DimAdd, DimMul)):
            a_hi = _dim_upper_bound(d.args[0])
            b_hi = _dim_upper_bound(d.args[1])
            if a_hi is None or b_hi is None:
                return None
            if isinstance(target, DimMin):
                return min(a_hi, b_hi)
            if isinstance(target, DimMax):
                return max(a_hi, b_hi)
            if isinstance(target, DimAdd):
                return a_hi + b_hi
            return None if a_hi < 0 or b_hi < 0 else a_hi * b_hi  # DimMul
        if isinstance(target, (DimFloorDiv, DimMod)):
            a_hi = _dim_upper_bound(d.args[0])
            b_val = _static_dim_value(d.args[1])
            if a_hi is None or b_val is None or b_val <= 0:
                return None
            return a_hi // b_val if isinstance(target, DimFloorDiv) else b_val - 1
    return None


def _canonical_shard(sl: "ShardLayout", out_shape) -> "ShardLayout":
    """Canonical output ``ShardLayout`` for a replicated input (nothing for the
    generic propagator to carry): C-order strides over ``out_shape``, all-ones
    when the shape is non-static; ``attrs`` and ``mesh`` pass through.
    """
    out_shape = tuple(out_shape)
    strides = try_c_order_strides(out_shape) or tuple(1 for _ in out_shape)
    return ShardLayout(
        layout=Layout(shape=out_shape, strides=strides),
        attrs=sl.attrs,
        mesh=sl.mesh,
    )


@register_typeinfer(TopK)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
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
    if not is_dim_expr(k):
        ctx.error(
            call, f"k must be an int, DimVar, or dim expression, got {type(k).__name__}"
        )
    k_static = _static_dim_value(k)
    if k_static is not None and k_static < 0:
        ctx.error(call, f"k must be non-negative, got {k_static}")
    axis_len = x_ty.shape[axis]
    if k_static is not None:
        if isinstance(axis_len, int) and k_static > axis_len:
            ctx.error(call, f"k={k_static} exceeds axis {axis} length {axis_len}")
    elif isinstance(axis_len, int):
        # Symbolic k: only a static axis length lets us check anything, and
        # even then only when k's upper bound is itself statically derivable
        # (see ``_dim_upper_bound``) — an unresolved bound fails open rather
        # than block a legitimate dynamic k.
        k_hi = _dim_upper_bound(k)
        if k_hi is not None and k_hi > axis_len:
            ctx.error(
                call,
                f"k's statically-derivable upper bound {k_hi} exceeds axis "
                f"{axis} length {axis_len}",
            )
    if isinstance(x_ty.layout, ShardLayout):
        la2ta = layout_axis_to_tensor_axis(x_ty.layout.layout.shape, x_ty.shape)
        if any(
            isinstance(a, Split) and la2ta[a.axis] == axis
            for a in x_ty.layout.attrs
        ):
            ctx.error(call, f"selected axis {axis} must not be Split-sharded")
    # The selected index is not recoverable from a partial (per-shard)
    # reduction.
    reject_partials(ctx, call, "x", x_ty.layout)
    out_shape = list(x_ty.shape)
    out_shape[axis] = k
    out_shape = tuple(out_shape)
    # A sharded input keeps its non-selected splits; the selected axis shrinks
    # to k. Derive the output layout instead of passing it through, so the
    # shard invariant size(layout) == size(shape) holds.
    new_layout = x_ty.layout
    if isinstance(x_ty.layout, ShardLayout):
        relation = build_relation(call, (x_ty,), ctx)
        derived = derive_output_shard_layout(
            (x_ty,), relation, out_shape, fresh_strides=True
        )
        # A replicated input has no Split/Partial for the propagator to carry
        # (derive returns None); still shrink the selected axis so the layout
        # keeps size parity with the output shape.
        new_layout = (
            derived if derived is not None else _canonical_shard(x_ty.layout, out_shape)
        )
    values_ty = TensorType(
        shape=out_shape, dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage
    )
    indices_ty = TensorType(
        shape=out_shape, dtype=DType.i64, layout=new_layout, storage=x_ty.storage
    )
    return TupleType(fields=(values_ty, indices_ty))


@register_type_relation(TopK)
def _topk_type_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward relation for shard propagation. Every non-selected axis projects
    to itself, so its sharding carries through unchanged. The selected axis is a
    fresh, data-dependent selection (not a view of the input axis), so it does
    not project — the output extent is synthesized from the shrunk output shape.
    """
    (x,) = input_types
    rank = len(x.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    in_dims = ", ".join(f"d{i}" for i in range(rank))
    out_dims = ", ".join("0" if i == axis else f"d{i}" for i in range(rank))
    in_map = isl.map(f"{{ [{in_dims}] -> [{in_dims}] }}")
    out_map = isl.map(f"{{ [{in_dims}] -> [{out_dims}] }}")
    return AccessRelationResult(domain=build_domain(x.shape), maps=(in_map, out_map))

@register_access_relation(TopK)
def _topk_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL level.

    The reduction axis is data-dependent (top-k indices come from sort), so the
    input access relation is an isl.map "scans the whole axis" rather than a
    multi_aff. Output values/indices are leading-dims identity with a new
    independent topk axis.
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    in_dims = ", ".join(f"i{i}" for i in range(rank))
    out_dims = ", ".join(f"i{i}" if i != axis else "j" for i in range(rank))
    # Input relation: every output position [.., j, ..] depends on the entire
    # axis range of the input. Express as map dropping the axis dim.
    leading = ", ".join(f"i{i}" for i in range(rank) if i != axis)
    if leading:
        in_rel = isl.map(f"{{ [{out_dims}] -> [{in_dims}] }}")
    else:
        in_rel = isl.map(f"{{ [j] -> [i{axis}] }}")
    # Output identity: trivial map from output to itself.
    out_id = isl.multi_aff(f"{{ [{out_dims}] -> [{out_dims}] }}")
    return AccessRelations(inputs=(in_rel,), outputs=(out_id, out_id))


def _local_dim_bindings(x: "TensorValue") -> dict:
    """``DimVar`` name → concrete size bindings recoverable from ``x`` alone:
    ``x.type.shape`` is the (possibly symbolic) type-level shape and
    ``x.data.shape`` its concrete runtime counterpart, so zipping the two
    binds every axis where the type carries a bare ``DimVar``. Mirrors
    ``evaluator.interpreter._bind_dim_vars`` but scoped to this one Call's
    sole tensor operand — ``k`` has no other operand to bind against, and in
    every context-length-derived-k case the same ``DimVar`` that sizes ``k``
    also sizes the axis being selected on, so it is reachable this way. A
    ``DimVar`` inside ``k`` that appears only in some *other* argument of the
    enclosing Function (not ``x``) is out of reach here: fixing that would
    need the interpreter's full ``dim_env`` threaded into ``EvalContext``,
    which this Call-local handler does not have access to.
    """
    bindings: dict = {}
    for axis, dim in enumerate(x.type.shape):
        if isinstance(dim, DimVar) and axis < len(x.data.shape):
            bindings[dim.name] = int(x.data.shape[axis])
    return bindings


@register_eval(TopK)
def _eval_topk(ctx):
    x = ctx.args[0]
    k = resolve_dim(ctx.op.k, _local_dim_bindings(x))
    vals, idx = torch.topk(
        x.data, k, dim=ctx.op.axis,
        largest=ctx.op.largest, sorted=ctx.op.sorted,
    )
    return TupleValue(
        elements=(
            TensorValue(
                data=vals.to(to_torch_dtype(ctx.result_type.fields[0].dtype)),
                type=ctx.result_type.fields[0],
            ),
            TensorValue(
                data=idx.to(to_torch_dtype(ctx.result_type.fields[1].dtype)),
                type=ctx.result_type.fields[1],
            ),
        )
    )


__all__ = ["TopK"]
