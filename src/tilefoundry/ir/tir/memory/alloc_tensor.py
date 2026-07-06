"""TIR-owned Expr Op: `tir.memory.AllocTensor`.

Introduces a storage root. Must be anchored by
a ``LetStmt`` to get positional identity (Expr DAG sharing would otherwise
merge distinct allocations).

The single attribute is the resulting ``TensorType`` (shape, dtype, layout,
storage). ``typeinfer`` returns it verbatim.
"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType


@register_op(name="alloc_tensor")
class AllocTensor(Op):
    """Allocate a tensor; the result type is carried on ``tensor_type``.

    Spec: tir.md §3.1

    Value form — the result ``Var`` MUST be anchored by ``LetStmt.value``.
    """
    tensor_type = ParamDef(kind="attribute", annotation=TensorType)
@register_typeinfer(AllocTensor)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return call.target.tensor_type
