"""Top-K selection HIR primitive.

SGLang baseline kernel K11 (TopK Gating Softmax) emits both the top-k values
and their indices. We model just the selection step here; the upstream Softmax
remains a separate node.

"""
from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.shard import Layout
from tilefoundry.ir.types.shard.shard_layout import (
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
)
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    AccessRelations,
    build_relation,
    register_access_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain
from tilefoundry.visitor_registry.shard_propagate import (
    _c_order,
    derive_output_shard_layout,
)


@register_op
class TopK(Op):
    """Multi-output (values, indices). Reduces the ``axis`` dim to length ``k``.

    ``largest`` selects the greatest (vs smallest) elements; ``sorted`` requests
    the selected elements be returned in order. Indices are ``i64``.
    """
    x = ParamDef(kind="input", pattern=Tensor)
    k = ParamDef(kind="attribute", annotation=int)
    axis = ParamDef(kind="attribute", annotation=int, default=-1)
    largest = ParamDef(kind="attribute", annotation=bool, default=True)
    sorted = ParamDef(kind="attribute", annotation=bool, default=True)


def _canonical_shard(sl: "ShardLayout", out_shape) -> "ShardLayout":
    """Canonical output ``ShardLayout`` for a replicated input (nothing for the
    generic propagator to carry): C-order strides over ``out_shape``, all-ones
    when the shape is non-static; ``attrs`` and ``mesh`` pass through.
    """
    out_shape = tuple(out_shape)
    strides = _c_order(out_shape) or tuple(1 for _ in out_shape)
    return ShardLayout(
        layout=Layout(shape=out_shape, strides=strides),
        attrs=sl.attrs,
        mesh=sl.mesh,
    )


@register_typeinfer(TopK)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        raise TypeError("TopK: x must be at least rank-1")
    rank = len(x_ty.shape)
    axis = call.target.axis
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise TypeError(f"TopK: axis {call.target.axis} out of range for rank {rank}")
    if call.target.k < 0:
        raise TypeError(f"TopK: k must be non-negative, got {call.target.k}")
    axis_len = x_ty.shape[axis]
    if isinstance(axis_len, int) and call.target.k > axis_len:
        raise TypeError(
            f"TopK: k={call.target.k} exceeds axis {axis} length {axis_len}"
        )
    if isinstance(x_ty.layout, ShardLayout):
        la2ta = layout_axis_to_tensor_axis(x_ty.layout.layout.shape, x_ty.shape)
        if any(
            isinstance(a, Split) and la2ta[a.axis] == axis
            for a in x_ty.layout.attrs
        ):
            raise TypeError(f"TopK: selected axis {axis} must not be Split-sharded")
    out_shape = list(x_ty.shape)
    out_shape[axis] = call.target.k
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

@register_eval(TopK)
def _eval_topk(ctx):
    vals, idx = torch.topk(
        ctx.args[0].data, ctx.op.k, dim=ctx.op.axis,
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
