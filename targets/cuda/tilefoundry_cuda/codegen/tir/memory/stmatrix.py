"""Refuse stmatrix emission until runtime support exists."""

from __future__ import annotations

from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_cuda import CudaTarget
from tilefoundry_cuda.codegen.context import CudaCodegenContext
from tilefoundry_cuda.ir.tir.memory.stmatrix import StMatrix


@register_codegen(CudaTarget, Role.EMIT, StMatrix)
def _emit(call, ctx: CudaCodegenContext) -> None:
    raise RuntimeError("tir.cuda.memory.StMatrix: no emitter without stmatrix runtime support")
