"""Emitter for ``tir.nn.RMSNorm`` (fused RMS normalization stmt).

Emits one call to ``tilefoundry::ops::rmsnorm(src, dst, weight, eps)``; the
row-wise reduction and rescaling live in the runtime header.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.nn.rms_norm import RMSNorm


@register_codegen_cuda(RMSNorm)
def _emit(call, ctx: CodegenContext) -> None:
    src, dst, weight = call.args[0], call.args[1], call.args[2]
    if not isinstance(src, Var) or not isinstance(dst, Var) or not isinstance(weight, Var):
        raise RuntimeError("tir.nn.RMSNorm: demo path expects Var operands for src/dst/weight")
    src_name = ctx.name_for(src)
    dst_name = ctx.name_for(dst)
    weight_name = ctx.name_for(weight)

    eps = call.target.eps
    ctx.emit(f"tilefoundry::ops::rmsnorm({src_name}, {dst_name}, {weight_name}, {eps}f);")
