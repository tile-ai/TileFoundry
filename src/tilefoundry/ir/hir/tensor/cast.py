from __future__ import annotations

import isl

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    build_relation,
    identity_relations,
    register_access_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.relation_build import build_domain
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout


@register_op
class Cast(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    dtype = ParamDef(kind="attribute", annotation=DType)


# GLOBAL-level: only the dtype changes; input/output are elementwise identity.
register_access_relation(Cast)(identity_relations(1))


@register_type_relation(Cast)
def _cast_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward relation for Cast: identity — only the dtype changes, shape and
    layout pass through unchanged."""
    (x,) = input_types
    dims = [f"d{i}" for i in range(len(x.shape))]
    src = "[" + ", ".join(dims) + "]"
    ident = isl.map(f"{{ {src} -> [{', '.join(dims)}] }}")
    return AccessRelationResult(domain=build_domain(x.shape), maps=(ident, ident))


@register_typeinfer(Cast)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    new_layout = x_ty.layout
    if isinstance(x_ty.layout, ShardLayout):
        relation = build_relation(call, (x_ty,), ctx)
        derived = derive_output_shard_layout((x_ty,), relation, x_ty.shape)
        if derived is not None:
            new_layout = derived
    return TensorType(
        shape=x_ty.shape, dtype=call.target.dtype, layout=new_layout, storage=x_ty.storage
    )


@register_eval(Cast)
def _eval_cast(ctx):

    out = ctx.args[0].data.to(to_torch_dtype(ctx.op.dtype))
    return TensorValue(data=out, type=ctx.result_type)
