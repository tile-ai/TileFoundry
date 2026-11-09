"""Emitters for the wgmma group and async proxy fence TIR ops, each one PTX instruction."""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir import expr
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context
from tilefoundry_nvidia.cuda.ir.tir import sync


@codegen.register(target.Target, codegen.Role.EMIT, sync.WgmmaFence)
def emit_fence(call, ctx: context.Context) -> None:
    ctx.emit('asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");')


@codegen.register(target.Target, codegen.Role.EMIT, sync.WgmmaCommitGroup)
def emit_commit(call, ctx: context.Context) -> None:
    ctx.emit('asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");')


@codegen.register(target.Target, codegen.Role.EMIT, sync.WgmmaWaitGroup)
def emit_wait(call, ctx: context.Context) -> None:
    n = call.target.n
    ctx.emit(f'asm volatile("wgmma.wait_group.sync.aligned %0;\\n" :: "n"({n}) : "memory");')


@codegen.register(target.Target, codegen.Role.EMIT, sync.FenceProxyAsync)
def emit_proxy_fence(call, ctx: context.Context) -> None:
    space = "global" if call.target.storage is expr.StorageKind.GMEM else "shared::cta"
    ctx.emit(f'asm volatile("fence.proxy.async.{space};\\n" ::: "memory");')


__all__: list[str] = []
