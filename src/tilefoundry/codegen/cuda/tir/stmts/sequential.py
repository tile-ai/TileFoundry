"""Emitter for ``tir.Sequential`` — packs a list of Stmts in order."""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, Sequential)
def _emit(node: Sequential, ctx: CudaCodegenContext) -> None:
    for stmt in node.body:
        ctx.emit_node(stmt)
