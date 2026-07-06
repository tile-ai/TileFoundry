"""HIR ``zeros(shape, dtype, storage)`` callable Op.

Allocate a zero-initialised tensor with explicit ``shape`` / ``dtype``
/ ``storage``. Sharding is set up later by a separate ``reshard``
call — ``zeros`` is intentionally storage-only, no layout attribute.
"""
from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TensorType


@register_op
class Zeros(Op):
    """Allocate a zero-initialised tensor with the given shape / dtype / storage.

    Spec: hir.md §2.2
    """
    shape = ParamDef(kind="attribute", annotation=tuple)
    dtype = ParamDef(kind="attribute", annotation=DType)
    storage = ParamDef(kind="attribute", default=StorageKind.GMEM)

@register_typeinfer(Zeros)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    op = call.target
    return TensorType(
        shape=op.shape, dtype=op.dtype, layout=None, storage=op.storage
    )


@register_eval(Zeros)
def _eval_zeros(ctx):


    shape = tuple(int(d) for d in ctx.op.shape)
    data = torch.zeros(shape, dtype=to_torch_dtype(ctx.op.dtype), device=ctx.device)
    return TensorValue(data=data, type=ctx.result_type)
