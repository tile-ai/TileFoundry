from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    build_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class Transpose(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    perm = ParamDef(kind="attribute", annotation=tuple)


@register_type_relation(Transpose)
def _transpose_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward relation for Transpose: an identity input map and an output map
    that permutes the iteration dims (output axis ``o`` reads domain dim
    ``perm[o]``). The shard engine reorders the input's layout positions by their
    owning tensor axis, preserving any factorization."""
    (x,) = input_types
    perm = call.target.perm
    rank = len(x.shape)
    dims = [f"d{i}" for i in range(rank)]
    src = "[" + ", ".join(dims) + "]"
    in_map = isl.map(f"{{ {src} -> [{', '.join(dims)}] }}")
    out_map = isl.map(f"{{ {src} -> [{', '.join(dims[perm[o]] for o in range(rank))}] }}")
    return AccessRelationResult(domain=build_domain(x.shape), maps=(in_map, out_map))


@register_typeinfer(Transpose)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    perm = call.target.perm
    if len(perm) != len(x_ty.shape):
        ctx.error(call, f"perm length {len(perm)} != rank {len(x_ty.shape)}")
    new_shape = tuple(x_ty.shape[p] for p in perm)

    new_layout = x_ty.layout
    if isinstance(x_ty.layout, ShardLayout):
        relation = build_relation(call, (x_ty,), ctx)
        derived = derive_output_shard_layout(
            (x_ty,), relation, new_shape, fresh_strides=False
        )
        if derived is not None:
            new_layout = derived
    return TensorType(
        shape=new_shape, dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage
    )


@register_eval(Transpose)
def _eval_transpose(ctx):
    out = torch.permute(ctx.args[0].data, tuple(ctx.op.perm)).contiguous()
    return TensorValue(data=out, type=ctx.result_type)
