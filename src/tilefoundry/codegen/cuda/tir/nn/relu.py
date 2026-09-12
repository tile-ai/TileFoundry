"""Emitter for ``tir.nn.ReLU`` (stmt-form pointwise ReLU).

Emits the one pointwise runtime entry with the ``relu_op`` tag as its ``fn``:
``tilefoundry::ops::elementwise(dst, relu_op{}, src)``. The element count is
the destination layout's, and the loop lives in the runtime header. The
destination tensor must already have been materialised by a preceding
``LetStmt`` on an ``AllocTensor`` Expr Op.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.nn import ReLU
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, ReLU)
def _emit(call, ctx: CudaCodegenContext) -> None:
    src, dst = call.args[0], call.args[1]
    if not isinstance(src, Var) or not isinstance(dst, Var):
        raise RuntimeError("tir.nn.ReLU: demo path expects Var operands on both sides")
    src_name = ctx.name_for(src)
    dst_name = ctx.name_for(dst)
    ctx.emit(
        f"tilefoundry::ops::elementwise({dst_name}, "
        f"tilefoundry::primitive::relu_op{{}}, {src_name});"
    )
