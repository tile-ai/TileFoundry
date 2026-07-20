"""Per-token-group FP8 quantization op.

SGLang baseline uses ``per_token_group_quant_8bit_kernel`` for K01/K07/K14/K17.
This op covers the attention-path K01/K07.


Semantics: split the last axis into groups of size ``group``; for each group
emit a fp8 vector (same shape as the group) and a single f32 scale.

Output is a tuple ``(x_q, x_scale)``:

- ``x_q.shape == x.shape``, dtype = ``DType.fp8e4m3``.
- ``x_scale.shape == x.shape[:-1] + (x.shape[-1] // group,)``, dtype =
  ``DType.f32``.
"""
from __future__ import annotations

import isl

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    register_access_relation,
)
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op
class Quant(Op):
    """Per-token-group FP8 quantize. Multi-output (x_q, x_scale)."""
    x = ParamDef(kind="input", pattern=Tensor)
    scheme = ParamDef(kind="attribute", annotation=str, default="per_token_group")
    group = ParamDef(kind="attribute", annotation=int, default=128)
    target_dtype = ParamDef(kind="attribute", annotation=DType, default=DType.fp8e4m3)
@register_typeinfer(Quant)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        raise TypeError("Quant: x must be at least rank-1")
    for axis, reduction in enumerate(partial_reductions_by_axis(x_ty.layout)):
        if reduction is not None:
            raise TypeError(
                f"Quant: Partial input on x is unsound: x carries Partial({reduction}) "
                f"on mesh axis {axis}; "
                "per-group normalization does not commute. Insert reshard(x, "
                "Broadcast) before this consumer"
            )
    last = x_ty.shape[-1]
    group = call.target.group
    # Static divisibility check when last dim is a Python int.
    if isinstance(last, int):
        if last % group != 0:
            raise TypeError(
                f"Quant: last dim {last} not divisible by group={group}"
            )
        scale_last = last // group
    else:
        # Symbolic last dim: leave it to downstream verify; produce same Expr.
        scale_last = last  # placeholder; fine for current baseline (all dims int)
    x_q_ty = TensorType(
        shape=x_ty.shape,
        dtype=call.target.target_dtype,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )
    scale_ty = TensorType(
        shape=x_ty.shape[:-1] + (scale_last,),
        dtype=DType.f32,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )
    return TupleType(fields=(x_q_ty, scale_ty))

# ── Access relation (GLOBAL level) ────────────────────────────────────

@register_access_relation(Quant)
def _quant_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL black-box quant.

    - input ``x`` is read element-wise → identity multi_aff over the rank-N
      domain.
    - output ``x_q`` is element-wise identity (same shape).
    - output ``x_scale`` reduces over the in-group offset (last dim divided by
      ``group``); expressed as an isl map ``[..., j] -> [..., j // group]``.
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    group = call.target.group

    # Build identity over rank-N: { [i0,..,iN-1] -> [i0,..,iN-1] }.
    dims = ", ".join(f"i{k}" for k in range(rank))
    ident = isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")

    # Build per-group reduction map for x_scale: floor(last / group).
    if rank == 0:
        # Defensive: typeinfer above already rejects rank-0.
        scale_rel = ident  # pragma: no cover
    else:
        outer = ", ".join(f"i{k}" for k in range(rank - 1))
        last = f"i{rank - 1}"
        out_dims = (outer + ", ") if outer else ""
        scale_rel = isl.map(
            f"{{ [{dims}] -> [{out_dims}floor({last}/{group})] }}"
        )

    return AccessRelations(
        inputs=(ident,),
        outputs=(ident, scale_rel),
    )

__all__ = ["Quant"]
