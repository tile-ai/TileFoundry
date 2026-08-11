from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.layout_algebra import prefix_product
from tilefoundry.ir.types.shard.shard_layout import (
    Partial,
    ShardLayout,
    Split,
    split_target_axes,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelations,
    register_access_relation,
)


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")


@register_op(name="index_select")
class IndexSelect(Op):
    """Select whole slices along ``dim`` using a 1-D integer ``index``."""

    x = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="input", pattern=Tensor)
    dim = ParamDef(kind="attribute", annotation=int, default=0)


def _norm_dim(dim: int, rank: int, ctx=None, call=None) -> int:
    normalized = dim + rank if dim < 0 else dim
    if normalized < 0 or normalized >= rank:
        msg = f"dim {dim} out of range for rank {rank}"
        if ctx is not None:
            ctx.error(call, msg)
        raise ValueError(f"IndexSelect: {msg}")
    return normalized


def _index_select_shard_layout(call, ctx, x_ty, dim: int, out_shape: tuple):
    """Derive a natural contiguous shard layout for a whole-slice selection.

    Broadcast and Partial states carry through. A Split on the selected dim
    becomes ``Partial(sum)``. Composed or ambiguous multi-Split layouts fail
    closed.
    """
    sl = x_ty.layout
    if not isinstance(sl, ShardLayout):
        return sl
    if not isinstance(sl.layout, Layout):
        ctx.error(
            call,
            f"dim {dim} has a composed shard layout; cannot derive an output layout",
        )
    targets = split_target_axes(sl, tuple(x_ty.shape))
    splits = [(i, t) for i, t in enumerate(targets) if t is not None]
    on_dim = [i for i, target in splits if target == dim]
    if on_dim and len(splits) > 1:
        ctx.error(
            call,
            f"dim {dim} index_select over a shard layout with multiple Split "
            "axes including the selected dim; cannot derive an output layout",
        )
    natural = Layout(shape=out_shape, strides=prefix_product(out_shape))
    if on_dim:
        mesh_idx = on_dim[0]
        new_attrs = tuple(
            Partial(reduction="sum") if i == mesh_idx else a for i, a in enumerate(sl.attrs)
        )
        return ShardLayout(layout=natural, attrs=new_attrs, mesh=sl.mesh)

    new_attrs = tuple(
        Split(axis=target) if isinstance(attr, Split) else attr
        for attr, target in zip(sl.attrs, targets)
    )
    return ShardLayout(layout=natural, attrs=new_attrs, mesh=sl.mesh)


@register_typeinfer(IndexSelect)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    index_ty = ctx.type_of(call.args[1])
    if len(index_ty.shape) != 1:
        ctx.error(call, f"index must be 1-D, got shape {index_ty.shape}")
    if index_ty.dtype not in (DType.i32, DType.i64):
        ctx.error(call, f"index must have dtype i32 or i64, got {index_ty.dtype}")
    dim = _norm_dim(call.target.dim, len(x_ty.shape), ctx, call)

    new_shape = list(x_ty.shape)
    new_shape[dim] = index_ty.shape[0]
    new_layout = _index_select_shard_layout(call, ctx, x_ty, dim, tuple(new_shape))
    return TensorType(
        shape=tuple(new_shape), dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage
    )


@register_access_relation(IndexSelect)
def _index_select_access_relation(call: "Call", ctx) -> AccessRelations:
    """GLOBAL level: IndexSelect pulls per-index slices.

    The access pattern is data-dependent on the index arg, so input data is
    OPAQUE while the index and output relations are identities.
    """
    idx_rank = len(ctx.type_of(call.args[1]).shape)
    out_rank = len(ctx.type_of(call).shape)
    return AccessRelations(
        inputs=(OPAQUE, _identity(idx_rank)),
        outputs=(_identity(out_rank),),
    )


@register_eval(IndexSelect)
def _eval_index_select(ctx):
    x = ctx.args[0].data
    index = ctx.args[1].data
    dim = _norm_dim(ctx.op.dim, x.dim())
    return TensorValue(
        data=torch.index_select(x, dim, index),
        type=ctx.result_type,
    )


__all__ = ["IndexSelect"]
