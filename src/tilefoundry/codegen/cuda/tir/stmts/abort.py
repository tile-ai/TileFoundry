"""Emit the TIR abort effect."""

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.abort import Abort
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, Abort)
def _emit(node: Abort, ctx: CudaCodegenContext) -> None:
    ctx.emit("__trap();")
