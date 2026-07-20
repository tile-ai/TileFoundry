from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shape_helpers import i64_const
from tilefoundry.visitor_registry import register_typeinfer


@register_op
class Stack(Op):
    """Variadic input op. See Concat for encoding rationale."""
    is_variadic: ClassVar[bool] = True

    inputs = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
@register_typeinfer(Stack)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    if not call.args:
        ctx.error(call, "Stack requires at least one input")
    types = [ctx.type_of(a) for a in call.args]
    base = types[0]
    for t in types[1:]:
        if t.shape != base.shape:
            ctx.error(call, "Stack inputs must have identical shape")
        if t.dtype != base.dtype:
            ctx.error(call, "Stack inputs must have matching dtype")
    axis = call.target.axis
    new_len = i64_const(len(call.args))
    new_shape = list(base.shape)
    new_shape.insert(axis, new_len)
    storage = resolve_anchor_storage(ctx, call, *(t.storage for t in types))
    return TensorType(
        shape=tuple(new_shape), dtype=base.dtype, layout=base.layout, storage=storage
    )
