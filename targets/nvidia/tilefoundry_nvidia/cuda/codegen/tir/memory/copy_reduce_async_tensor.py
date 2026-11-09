"""Refuse tensor-map reduction emission until runtime support exists."""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context
from tilefoundry_nvidia.cuda.ir.tir import memory


@codegen.register(target.Target, codegen.Role.EMIT, memory.CopyReduceAsyncTensor)
def emit(call, ctx: context.Context) -> None:
    raise RuntimeError(
        "tir.cuda.memory.CopyReduceAsyncTensor: no emitter without host-encoded tensor maps"
    )


__all__: list[str] = []
