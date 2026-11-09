"""TIR view Expr Op: `tir.memory.Span`.

Describes a contiguous memory span over a pointer.
Placeholder — demo does not use it; typeinfer mirrors input type.
"""

from __future__ import annotations

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, pattern, types


@expr.register_op(name="memory_span", dialect='T', category='memory')
class Span(expr.Op):
    """Re-interpret a memory region as a typed tensor (value form)."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


@typeinfer.register(Span)
def _(call: "Call", ctx: "TypeInferContext") -> types.TensorType:
    return ctx.type_of(call.args[0])


__all__ = ["Span"]
