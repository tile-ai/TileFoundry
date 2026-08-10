"""Emitter for `tir.memory.Copy` — dispatches ``tilefoundry::copy`` or ``cute::copy``.

Emitter for `tir.memory.Copy` — dispatches ``tilefoundry::copy`` (when either
operand carries ``ShardLayout``) or ``cute::copy`` (plain↔plain).
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.types.shape_helpers import shape_has_dim_var, shape_runtime_total
from tilefoundry.ir.types.shard.shard_layout import ShardLayout


def _is_shard(var) -> bool:
    return isinstance(getattr(var.type, "layout", None), ShardLayout)


def _tensor_expr(var, ctx: CodegenContext) -> str:
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


def _has_dyn_shape(var) -> bool:
    shape = getattr(getattr(var, "type", None), "shape", ())
    return shape_has_dim_var(shape)


@register_codegen_cuda(Copy)
def _emit(call, ctx: CodegenContext) -> None:
    source, destination = call.args[0], call.args[1]
    src = _tensor_expr(source, ctx)
    dst = _tensor_expr(destination, ctx)
    src_shard = _is_shard(source)
    dst_shard = _is_shard(destination)



    if not src_shard and not dst_shard and (
        _has_dyn_shape(source) or _has_dyn_shape(destination)
    ):
        N = shape_runtime_total(
            destination.type.shape, ctx._dim_var_runtime,
        )
        ctx.emit(f"tilefoundry::ops::copy_n({src}, {dst}, {N});")
        return
    if src_shard or dst_shard:
        ctx.emit(f"tilefoundry::copy({src}, {dst});")
    else:
        ctx.emit(f"cute::copy({src}, {dst});")
