"""Pure-value whole-slice indexed copy."""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.tensor.index_add import _infer_index_write
from tilefoundry.ir.hir.tensor.index_select import _norm_dim
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry import register_typeinfer


@register_op(name="index_copy")
class IndexCopy(Op):
    """Return ``dst`` with ``src`` slices copied to ``index`` along ``dim``."""

    dst = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="input", pattern=Tensor)
    src = ParamDef(kind="input", pattern=Tensor)
    dim = ParamDef(kind="attribute", annotation=int, default=0)


@register_typeinfer(IndexCopy)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return _infer_index_write(
        call,
        ctx,
        op_name="IndexCopy",
        index_dtypes=(DType.i64,),
    )


@register_eval(IndexCopy)
def _eval_index_copy(ctx):
    dst, index, src = (arg.data for arg in ctx.args)
    dim = _norm_dim(ctx.op.dim, dst.dim())
    return TensorValue(
        data=dst.clone().index_copy_(dim, index, src),
        type=ctx.result_type,
    )


__all__ = ["IndexCopy"]
