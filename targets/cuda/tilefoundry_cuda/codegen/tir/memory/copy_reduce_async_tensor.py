"""Refuse tensor-map reduction emission until runtime support exists."""

from __future__ import annotations

from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_cuda import CudaTarget
from tilefoundry_cuda.codegen.context import CudaCodegenContext
from tilefoundry_cuda.ir.tir.memory.copy_reduce_async_tensor import CopyReduceAsyncTensor


@register_codegen(CudaTarget, Role.EMIT, CopyReduceAsyncTensor)
def _emit(call, ctx: CudaCodegenContext) -> None:
    raise RuntimeError(
        "tir.cuda.memory.CopyReduceAsyncTensor: no emitter without host-encoded tensor maps"
    )
