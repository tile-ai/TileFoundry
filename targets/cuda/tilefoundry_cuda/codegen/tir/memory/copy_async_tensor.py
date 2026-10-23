"""Refuse tensor-map emission until host tensor-map construction exists."""

from __future__ import annotations

from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_cuda import CudaTarget
from tilefoundry_cuda.codegen.context import CudaCodegenContext
from tilefoundry_cuda.ir.tir.memory.copy_async_tensor import CopyAsyncTensor


@register_codegen(CudaTarget, Role.EMIT, CopyAsyncTensor)
def _emit(call, ctx: CudaCodegenContext) -> None:
    raise RuntimeError(
        "tir.cuda.memory.CopyAsyncTensor: no emitter without host-encoded tensor maps"
    )
