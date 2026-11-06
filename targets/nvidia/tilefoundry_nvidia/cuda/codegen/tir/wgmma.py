"""Emitters for the wgmma group and async proxy fence TIR ops, each one PTX instruction."""

from __future__ import annotations

from tilefoundry.ir.types import StorageKind
from tilefoundry.visitor_registry.registries import Role, register_codegen
from tilefoundry_nvidia.cuda import CudaTarget
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext
from tilefoundry_nvidia.cuda.ir.tir.sync.wgmma import (
    FenceProxyAsync,
    WgmmaCommitGroup,
    WgmmaFence,
    WgmmaWaitGroup,
)


@register_codegen(CudaTarget, Role.EMIT, WgmmaFence)
def _emit_fence(call, ctx: CudaCodegenContext) -> None:
    ctx.emit('asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");')


@register_codegen(CudaTarget, Role.EMIT, WgmmaCommitGroup)
def _emit_commit(call, ctx: CudaCodegenContext) -> None:
    ctx.emit('asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");')


@register_codegen(CudaTarget, Role.EMIT, WgmmaWaitGroup)
def _emit_wait(call, ctx: CudaCodegenContext) -> None:
    n = call.target.n
    ctx.emit(f'asm volatile("wgmma.wait_group.sync.aligned %0;\\n" :: "n"({n}) : "memory");')


@register_codegen(CudaTarget, Role.EMIT, FenceProxyAsync)
def _emit_proxy_fence(call, ctx: CudaCodegenContext) -> None:
    space = "global" if call.target.storage is StorageKind.GMEM else "shared::cta"
    ctx.emit(f'asm volatile("fence.proxy.async.{space};\\n" ::: "memory");')
