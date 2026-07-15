"""HIR generic Reduce op with kind enum."""

from __future__ import annotations

import isl

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    build_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout

__all__ = ["ReduceKind", "Reduce"]

@register_op
class Reduce(Op):
    """Axis reduction over ``x`` (``mean`` / ``sum`` / ``abs_max`` / ``max``)."""
    x = ParamDef(kind="input", pattern=Tensor)
    axes = ParamDef(kind="attribute", annotation=tuple)
    keepdim = ParamDef(kind="attribute", annotation=bool, default=True)
    kind = ParamDef(kind="attribute", annotation=ReduceKind, default=ReduceKind.MEAN)

def _reduced_axes(call: "Call", rank: int) -> tuple:
    return tuple(a % rank if a < 0 else a for a in call.target.axes)


@register_type_relation(Reduce)
def _reduce_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward relation for Reduce: an identity input map; the output map keeps
    every axis (keepdim) or drops the reduced axes (no keepdim). The reduced
    axes are reported as completely-reduced dims, so a Split on them collapses
    to Broadcast and their layout positions collapse to size 1.
    """
    (x,) = input_types
    rank = len(x.shape)
    reduced = _reduced_axes(call, rank)
    dims = [f"d{i}" for i in range(rank)]
    src = "[" + ", ".join(dims) + "]"
    in_map = isl.map(f"{{ {src} -> [{', '.join(dims)}] }}")
    out_dims = (
        dims if call.target.keepdim else [dims[i] for i in range(rank) if i not in reduced]
    )
    out_map = isl.map(f"{{ {src} -> [{', '.join(out_dims)}] }}")
    return AccessRelationResult(domain=build_domain(x.shape), maps=(in_map, out_map))


@register_typeinfer(Reduce)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    keepdim = call.target.keepdim
    rank = len(x_ty.shape)
    reduced = _reduced_axes(call, rank)

    new_shape = list(x_ty.shape)
    for a in sorted(reduced, reverse=True):
        if keepdim:
            new_shape[a] = 1
        else:
            new_shape.pop(a)
    out_shape = tuple(new_shape)

    new_layout = x_ty.layout
    if isinstance(x_ty.layout, ShardLayout):
        relation = build_relation(call, (x_ty,), ctx)
        derived = derive_output_shard_layout(
            (x_ty,),
            relation,
            out_shape,
            complete_reduction_dims=frozenset(reduced),
            fresh_strides=True,
        )
        if derived is not None:
            new_layout = derived

    return TensorType(
        shape=out_shape,
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


@register_eval(Reduce)
def _eval_reduce(ctx):
    x = ctx.args[0].data
    axes = tuple(ctx.op.axes)
    keepdim = ctx.op.keepdim
    kind = ctx.op.kind
    if kind is ReduceKind.MEAN:
        out = x.mean(dim=axes, keepdim=keepdim)
    elif kind is ReduceKind.SUM:
        out = x.sum(dim=axes, keepdim=keepdim)
    elif kind is ReduceKind.ABS_MAX:
        out = x.abs().amax(dim=axes, keepdim=keepdim)
    elif kind is ReduceKind.MAX:
        out = x.amax(dim=axes, keepdim=keepdim)
    else:
        raise ValueError(f"evaluator: unsupported ReduceKind {kind}")
    return TensorValue(data=out, type=ctx.result_type)
