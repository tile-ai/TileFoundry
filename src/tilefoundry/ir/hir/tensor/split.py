from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.visitor_registry import register_typeinfer


@register_op
class Split(Op):
    """Multi-output op. `Call.type` is `TupleType` (§2.9 / §6.3.1)."""
    x = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
    num_splits = ParamDef(kind="attribute", annotation=int)
@register_typeinfer(Split)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    x_ty = ctx.type_of(call.args[0])
    axis = call.target.axis
    n = call.target.num_splits
    orig = x_ty.shape[axis]
    v = static_dim_value(orig)
    if v is not None:
        if v % n != 0:
            ctx.error(call, f"axis {axis} extent {v} not divisible by {n}")
        part_len = v // n
    else:
        # Symbolic: keep the original dim Expr (coarse; tighter dim.* division
        # op can be added later).
        part_len = orig
    part_shape = list(x_ty.shape)
    part_shape[axis] = part_len
    part_ty = TensorType(
        shape=tuple(part_shape), dtype=x_ty.dtype, layout=x_ty.layout, storage=x_ty.storage
    )
    return TupleType(fields=tuple(part_ty for _ in range(n)))
