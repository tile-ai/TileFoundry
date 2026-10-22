"""Refuse stmatrix emission until runtime support exists."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.cuda.memory.stmatrix import StMatrix
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, StMatrix)
def _emit(call, ctx: CudaCodegenContext) -> None:
    raise RuntimeError("tir.cuda.memory.StMatrix: no emitter without stmatrix runtime support")
