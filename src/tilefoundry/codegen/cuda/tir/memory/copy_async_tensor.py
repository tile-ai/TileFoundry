"""Refuse tensor-map emission until host tensor-map construction exists."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.cuda.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, CopyAsyncTensor)
def _emit(call, ctx: CudaCodegenContext) -> None:
    raise RuntimeError(
        "tir.cuda.memory.CopyAsyncTensor: no emitter without host-encoded tensor maps"
    )
