"""Emit the TIR abort effect."""

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.abort import Abort


@register_codegen_cuda(Abort)
def _emit(node: Abort, ctx: CodegenContext) -> None:
    ctx.emit("assert(false);")
