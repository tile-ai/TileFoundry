from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import (
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
)


@register_op
class Gather(Op):
    """Gather along one axis."""

    x = ParamDef(kind="input", pattern=Tensor)
    indices = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)


def _norm_axis(axis: int, rank: int) -> int:
    a = axis + rank if axis < 0 else axis
    if a < 0 or a >= rank:
        raise TypeError(f"Gather: axis {axis} out of range for rank {rank}")
    return a


def _sliced_shard_layout(x_ty, axis: int, idx_shape: tuple):
    """Return the sliced ``ShardLayout`` for a pure single-index gather.

    A scalar index or rank-1 ``(1,)`` index on a non-sharded axis is a slice:
    scalar indices drop that axis's cute positions, while ``(1,)`` indices keep
    them as size-1 positions. Gather along a Split axis, multi-index gather, or
    composed layout passes the input layout through unchanged.
    """
    sl = x_ty.layout
    if not isinstance(sl, ShardLayout) or not isinstance(sl.layout, Layout):
        return sl
    # Only a scalar index or a rank-1 ``(1,)`` index is a pure slice; any other
    # form (incl. a total-size-1 multi-index like ``(1, 1)``) passes through.
    scalar_idx = idx_shape == ()
    if not scalar_idx and idx_shape != (1,):
        return sl
    cute_shape = sl.layout.shape
    cute_strides = sl.layout.strides
    if cute_strides is None or len(cute_strides) != len(cute_shape):
        return sl
    pos_to_axis = layout_axis_to_tensor_axis(cute_shape, tuple(x_ty.shape))
    if len(pos_to_axis) != len(cute_shape):
        return sl
    sliced = {p for p, a in enumerate(pos_to_axis) if a == axis}
    if not sliced:
        return sl
    # A gather along a ``Split`` (sharded) axis is not a pure slice; pass through.
    if any(isinstance(a, Split) and a.axis in sliced for a in sl.attrs):
        return sl
    if scalar_idx:
        # Drop the sliced positions; remap Split axes onto the survivors.
        keep = [p for p in range(len(cute_shape)) if p not in sliced]
        if not keep:
            return sl
        new_shape = tuple(cute_shape[p] for p in keep)
        new_strides = tuple(cute_strides[p] for p in keep)
        pos_map = {p: i for i, p in enumerate(keep)}
        new_attrs = tuple(
            Split(axis=pos_map[a.axis]) if isinstance(a, Split) else a
            for a in sl.attrs
        )
    else:
        # (1,)-style index: the axis survives at size 1.
        new_shape = tuple(1 if p in sliced else d for p, d in enumerate(cute_shape))
        new_strides = tuple(cute_strides)
        new_attrs = sl.attrs
    return ShardLayout(
        layout=Layout(shape=new_shape, strides=new_strides),
        attrs=new_attrs,
        mesh=sl.mesh,
    )


@register_typeinfer(Gather)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    idx_ty = ctx.type_of(call.args[1])
    axis = _norm_axis(call.target.axis, len(x_ty.shape))
    new_shape = list(x_ty.shape)
    # Replace axis-th dim with gathered indices' shape dims (flatten).
    new_shape = new_shape[:axis] + list(idx_ty.shape) + new_shape[axis + 1:]
    new_layout = _sliced_shard_layout(x_ty, axis, tuple(idx_ty.shape))
    return TensorType(
        shape=tuple(new_shape), dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage
    )


@register_eval(Gather)
def _eval_gather(ctx):
    x = ctx.args[0].data
    indices = ctx.args[1].data
    axis = _norm_axis(ctx.op.axis, x.dim())
    out = torch.index_select(x, axis, indices.reshape(-1).long())
    new_shape = x.shape[:axis] + tuple(indices.shape) + x.shape[axis + 1:]
    return TensorValue(data=out.reshape(new_shape), type=ctx.result_type)
