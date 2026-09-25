"""Codegen for TIR ``PtrOf`` — emits a pointer to the source tensor.

``PtrOf(src)`` returns the device pointer of *src*.  For a ShardTensor
source the pointer is read through ``.data()`` on the engine (the
backing cute tensor / gmem pointer).
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.tir.memory.ptr_of import PtrOf
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, PtrOf)
def _emit(let_stmt, ctx: CudaCodegenContext) -> None:
    call = let_stmt.value
    src = call.args[0]
    src_name = ctx.name_for(src)
    if ctx.is_kernel_param(src):
        src_name += "_tensor"
    var_name = ctx.name_for(let_stmt.var)
    ctx.emit(f"auto {var_name} = {src_name}.data();")
