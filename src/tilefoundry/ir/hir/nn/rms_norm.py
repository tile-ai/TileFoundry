"""Define fused last-axis RMS normalization.

The rank-one weight matches the final input extent; preceding dimensions pass
through, including symbolic expressions. Computation accumulates in f32 and
casts back to the input dtype, while the weight may use another dtype.
"""

from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    AccessRelations,
    register_access_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.isl_utility import to_domain


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")


@register_op(name="rms_norm")
class RMSNorm(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    weight = ParamDef(kind="input", pattern=Tensor)
    eps = ParamDef(kind="attribute", annotation=float, default=1e-6)


@register_typeinfer(RMSNorm)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    w_ty = ctx.type_of(call.args[1])

    if len(x_ty.shape) < 1:
        ctx.error(call, f"x must be rank ≥ 1, got shape {x_ty.shape}")
    if len(w_ty.shape) != 1:
        ctx.error(call, f"weight must be rank-1, got shape {w_ty.shape}")
    if x_ty.shape[-1] != w_ty.shape[0]:
        ctx.error(call, f"x last dim {x_ty.shape[-1]} != weight dim {w_ty.shape[0]}")

    for arg, ty in (("x", x_ty), ("weight", w_ty)):
        reject_partials(ctx, call, arg, ty.layout)

    return x_ty


@register_access_relation(RMSNorm)
def _rms_norm_relation(call: "Call", ctx) -> AccessRelations:
    """GLOBAL level: x identity, weight identity.

    GLOBAL level: x identity, weight identity (broadcast along last dim
    treated as identity at GLOBAL black-box; reduction is internal to the
    op).
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    return AccessRelations(
        inputs=(_identity(rank), _identity(1)),
        outputs=(_identity(rank),),
    )


@register_type_relation(RMSNorm)
def _rms_norm_type_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Model fused row-wise RMSNorm with batch axes as the domain.

    The reduced axis remains a range dimension shared by input, weight, and
    output maps. Local projection handles sharding before this relation.
    """
    x, weight = input_types
    x_shape, w_shape = x.shape, weight.shape
    if len(x_shape) < 1 or len(w_shape) != 1:
        raise NotImplementedError(
            "RMSNorm type_relation: x must be rank >= 1 and weight rank-1, "
            f"got x.shape={x_shape} weight.shape={w_shape}"
        )
    if x_shape[-1] != w_shape[0]:
        raise NotImplementedError(
            f"RMSNorm type_relation: x last dim {x_shape[-1]} != weight dim {w_shape[0]}"
        )
    reduce_extent = x_shape[-1]
    if not isinstance(reduce_extent, int) or isinstance(reduce_extent, bool):
        raise NotImplementedError(
            "RMSNorm type_relation: reduction axis must be a static int, "
            f"got {reduce_extent!r} -- a dynamic reduction axis has no isl "
            "representation here"
        )

    batch_shape = x_shape[:-1]
    domain, param_map = to_domain(batch_shape)
    dims = [f"d{i}" for i in range(len(batch_shape))]
    src = "[" + ", ".join(dims) + "]"
    row = ", ".join(dims + ["j"])
    row_map = isl.map(f"{{ {src} -> [{row}] : 0 <= j < {reduce_extent} }}")
    weight_map = isl.map(f"{{ {src} -> [j] : 0 <= j < {reduce_extent} }}")
    return AccessRelationResult(
        domain=domain, maps=(row_map, weight_map, row_map), param_map=param_map
    )


@register_eval(RMSNorm)
def _eval_rms_norm(ctx):

    xf = ctx.args[0].data.float()
    wf = ctx.args[1].data.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(ms + ctx.op.eps) * wf
    return TensorValue(data=out.to(to_torch_dtype(ctx.result_type.dtype)), type=ctx.result_type)
