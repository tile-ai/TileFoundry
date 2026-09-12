"""Emitter for `tir.Return`."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.stmts import Return
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, Return)
def _emit(node: Return, ctx: CudaCodegenContext) -> None:
    ctx.emit("return;")
