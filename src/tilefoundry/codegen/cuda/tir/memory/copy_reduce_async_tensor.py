"""Refuse tensor-map reduction emission until runtime support exists."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.cuda.memory.copy_reduce_async_tensor import CopyReduceAsyncTensor
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, CopyReduceAsyncTensor)
def _emit(call, ctx: CudaCodegenContext) -> None:
    raise RuntimeError(
        "tir.cuda.memory.CopyReduceAsyncTensor: no emitter without host-encoded tensor maps"
    )
