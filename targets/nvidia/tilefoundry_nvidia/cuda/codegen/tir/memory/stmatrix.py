"""Refuse stmatrix emission until runtime support exists."""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context
from tilefoundry_nvidia.cuda.ir.tir import memory


@codegen.register(target.Target, codegen.Role.EMIT, memory.StMatrix)
def emit(call, ctx: context.Context) -> None:
    raise RuntimeError("tir.cuda.memory.StMatrix: no emitter without stmatrix runtime support")


__all__: list[str] = []
