"""TIR-owned Expr Op: `tir.memory.AllocTensor`.

Introduces a storage root. Must be anchored by
a ``LetStmt`` to get positional identity (Expr DAG sharing would otherwise
merge distinct allocations).

The single attribute is the resulting ``TensorType`` (shape, dtype, layout,
storage). ``typeinfer`` returns it verbatim.
"""

from __future__ import annotations

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, types


@expr.register_op(name="alloc_tensor", dialect='T', category='memory')
class AllocTensor(expr.Op):
    """Allocate a tensor; the result type is carried on ``tensor_type``.

    Value form — the result ``Var`` MUST be anchored by ``LetStmt.value``.
    """

    tensor_type = expr.ParamDef(kind="attribute", annotation=types.TensorType)


@typeinfer.register(AllocTensor)
def _(call: "Call", ctx: "TypeInferContext") -> types.TensorType:
    return call.target.tensor_type


__all__ = ["AllocTensor"]
