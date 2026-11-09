"""Refuse tensor-map emission until host tensor-map construction exists."""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context
from tilefoundry_nvidia.cuda.ir.tir import memory


@codegen.register(target.Target, codegen.Role.EMIT, memory.CopyAsyncTensor)
@codegen.register(target.Target, codegen.Role.EMIT, memory.CopyAsyncTensorTx)
def emit(call, ctx: context.Context) -> None:
    raise RuntimeError(
        f"tir.cuda.memory.{type(call.target).__name__}: "
        "no emitter without host-encoded tensor maps"
    )


__all__: list[str] = []
