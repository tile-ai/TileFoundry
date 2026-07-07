"""TIR view Expr Op: `tir.view.MemorySpan`.

Describes a contiguous memory span over a pointer.
Placeholder — demo does not use it; typeinfer mirrors input type.
"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType


@register_op(name="memory_span")
class MemorySpan(Op):
    """Re-interpret a memory region as a typed tensor (value form)."""
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(MemorySpan)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return ctx.type_of(call.args[0])
