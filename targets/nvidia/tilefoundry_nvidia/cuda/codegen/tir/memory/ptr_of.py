"""Codegen for TIR ``PtrOf`` — emits a pointer to the source tensor.

``PtrOf(src)`` returns the device pointer of *src*.  For a ShardTensor
source the pointer is read through ``.data()`` on the engine (the
backing cute tensor / gmem pointer).
"""

from __future__ import annotations

from tilefoundry import codegen
from tilefoundry.ir.tir import memory
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context


@codegen.register(target.Target, codegen.Role.EMIT, memory.PtrOf)
def emit(let_stmt, ctx: context.Context) -> None:
    call = let_stmt.value
    src = call.args[0]
    src_name = ctx.name_for(src)
    if ctx.is_kernel_param(src):
        src_name += "_tensor"
    var_name = ctx.name_for(let_stmt.var)
    ctx.emit(f"auto {var_name} = {src_name}.data();")


__all__: list[str] = []
